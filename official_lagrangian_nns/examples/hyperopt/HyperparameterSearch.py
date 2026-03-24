import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map
import numpy as np # get rid of this eventually
import argparse
from jax import jit
from jax.experimental.ode import odeint
from functools import partial # reduces arguments to function by making some subset implicit
from lnn.jax_compat import optimizers, stax
import time
import sys
from lnn.core import lagrangian_eom_rk4, lagrangian_eom, raw_lagrangian_eom, unconstrained_eom
from lnn.models import mlp as make_mlp
from lnn.utils import wrap_coords

from ..double_pendulum.data import get_dataset, get_trajectory, get_trajectory_analytic
from ..double_pendulum.physics import analytical_fn

from jax.experimental.ode import odeint

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

# replace the lagrangian with a parameteric model
def learned_dynamics(params):
  @jit
  def dynamics(q, q_t):
#     assert q.shape == (2,)
    state = wrap_coords(jnp.concatenate([q, q_t]))
    return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
  return dynamics


from lnn.jax_compat import Dense, Relu, Softplus, Tanh, elementwise, serial


sigmoid = jit(lambda x: 1/(1+jnp.exp(-x)))
swish = jit(lambda x: x/(1+jnp.exp(-x)))
relu3 = jit(lambda x: jnp.clip(x, 0.0, float('inf'))**3)
Swish = elementwise(swish)
Relu3 = elementwise(relu3)

def extended_mlp(args):
    act = {
        'softplus': [Softplus, Softplus],
        'swish': [Swish, Swish],
        'tanh': [Tanh, Tanh],
        'tanh_relu': [Tanh, Relu],
        'soft_relu': [Softplus, Relu],
        'relu_relu': [Relu, Relu],
        'relu_relu3': [Relu, Relu3],
        'relu3_relu': [Relu3, Relu],
        'relu_tanh': [Relu, Tanh],
    }[args.act]
    hidden = args.hidden_dim
    output_dim = args.output_dim
    nlayers = args.layers
    
    layers = []
    layers.extend([
        Dense(hidden),
        act[0]
    ])
    for _ in range(nlayers - 1):
        layers.extend([
            Dense(hidden),
            act[1]
        ])
        
    layers.extend([Dense(output_dim)])
    
    return stax.serial(*layers)

vfnc = jax.jit(jax.vmap(analytical_fn))
vget = partial(jax.jit, backend='cpu')(jax.vmap(get_trajectory_analytic, (0, None), 0))
vget_unlimited = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic), (0, None), 0))

dataset_size=50
fps=10
samples=50



def new_get_dataset(rng, samples=1, t_span=[0, 10], fps=100, test_split=0.5, lookahead=1,
                    unlimited_steps=False, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs

    frames = int(fps*(t_span[1]-t_span[0]))
    times = jnp.linspace(t_span[0], t_span[1], frames)
    y0 = jnp.concatenate([
        jax.random.uniform(rng, (samples, 2))*2.0*np.pi,
        jax.random.uniform(rng+1, (samples, 2))*0.1
    ], axis=1)

    if not unlimited_steps:
        y = vget(y0, times)
    else:
        y = vget_unlimited(y0, times)
        
    #This messes it up!
#     y = np.concatenate(((y[..., :2]%(2*np.pi)) - np.pi, y[..., 2:]), axis=2)
    
    data['x'] = y[:, :-lookahead]
    data['dx'] = y[:, lookahead:] - data['x']
    data['x'] = jnp.reshape(data['x'], (-1, data['x'].shape[-1]))
    data['dx'] = jnp.reshape(data['dx'], (-1, data['dx'].shape[-1]))
    data['t'] = jnp.tile(times[:-lookahead], (samples,))

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx', 't']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def make_loss(args):
    objective = str(getattr(args, 'objective', 'delta'))
    if args.loss == 'l1':
        @jax.jit
        def gln_loss(params, batch, l2reg):
            state, targets = batch#_rk4
            leaves, _ = tree_flatten(params)
            l2_norm = sum(jnp.vdot(param, param) for param in leaves)
            if objective == 'xdot':
                preds = jax.vmap(partial(raw_lagrangian_eom, learned_dynamics(params)))(state)
            else:
                preds = jax.vmap(partial(lagrangian_eom_rk4, learned_dynamics(params), Dt=args.dt, n_updates=args.n_updates))(state)
            return jnp.sum(jnp.abs(preds - targets)) + l2reg*l2_norm/args.batch_size

    else:
        @jax.jit
        def gln_loss(params, batch, l2reg):
            state, targets = batch
            leaves, _ = tree_flatten(params)
            l2_norm = sum(jnp.vdot(param, param) for param in leaves)
            if objective == 'xdot':
                preds = jax.vmap(partial(raw_lagrangian_eom, learned_dynamics(params)))(state)
            else:
                preds = jax.vmap(partial(lagrangian_eom_rk4, learned_dynamics(params)))(state)
            return jnp.sum(jnp.square(preds - targets)) + l2reg*l2_norm/args.batch_size
        
            
    return gln_loss

from copy import deepcopy as copy
from tqdm import tqdm

def train(args, model, data, rng, checkpoint_callback=None, latest_checkpoint_callback=None):
    global opt_update, get_params, nn_forward_fn
    global best_params, best_loss, train_meta
    best_params = None
    best_loss = np.inf
    best_small_loss = np.inf
    best_iteration = -1
    stop_reason = 'completed'
    stop_iteration = int(args.num_epochs) - 1
    (nn_forward_fn, init_params) = model
    np_rng = np.random.default_rng(int(jax.random.randint(rng, (), 0, 2**31 - 1)))

    loss = make_loss(args)
    grad_clip = float(getattr(args, 'grad_clip', 0.0))
    lr_warmup_steps = max(0, int(getattr(args, 'lr_warmup_steps', 0)))
    param_check_every = max(1, int(getattr(args, 'param_check_every', 100)))
    latest_checkpoint_every = max(0, int(getattr(args, 'latest_checkpoint_every', 0)))

    def lr_schedule(t):
        t = jnp.asarray(t, dtype=jnp.float32)
        base_lr = jnp.asarray(args.lr, dtype=jnp.float32)
        low_lr = jnp.asarray(args.lr2, dtype=jnp.float32)
        if lr_warmup_steps > 0:
            warmup_progress = jnp.minimum((t + 1.0) / float(lr_warmup_steps), 1.0)
            stage1_lr = base_lr * warmup_progress
        else:
            stage1_lr = base_lr
        return jnp.where(t < (args.num_epochs // 2), stage1_lr, low_lr)

    opt_init, opt_update, get_params = optimizers.adam(lr_schedule)
    opt_state = opt_init(init_params)

    def params_are_finite(params):
        leaves, _ = tree_flatten(params)
        return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)

    @jax.jit
    def update_derivative(i, opt_state, batch, l2reg):
        params = get_params(opt_state)
        grads = jax.grad(loss, 0)(params, batch, l2reg)
        grad_leaves, _ = tree_flatten(grads)
        grad_norm_sq = sum(jnp.vdot(leaf, leaf) for leaf in grad_leaves)
        grad_norm = jnp.sqrt(grad_norm_sq)
        if grad_clip > 0.0:
            clip_scale = jnp.minimum(1.0, jnp.asarray(grad_clip, dtype=grad_norm.dtype) / (grad_norm + 1e-12))
            grads = tree_map(lambda g: g * clip_scale, grads)
        next_opt_state = opt_update(i, grads, opt_state)
        next_params = get_params(next_opt_state)
        return next_opt_state, next_params, grad_norm

    train_losses, test_losses = [], []
    target_key = 'xdot' if str(getattr(args, 'objective', 'delta')) == 'xdot' else 'dx'
    eval_batch_size = max(1, int(getattr(args, 'eval_batch_size', len(data['x']))))
    eval_train_samples = max(0, int(getattr(args, 'eval_train_samples', len(data['x']))))
    eval_test_samples = max(0, int(getattr(args, 'eval_test_samples', len(data['test_x']))))

    def pick_eval_indices(total: int, limit: int):
        if limit <= 0 or limit >= total:
            return None
        return np_rng.choice(total, size=limit, replace=False)

    eval_train_idx = pick_eval_indices(int(len(data['x'])), eval_train_samples)
    eval_test_idx = pick_eval_indices(int(len(data['test_x'])), eval_test_samples)

    def dataset_loss_batched(params, state_x, target_y):
        total = int(len(state_x))
        total_loss_sum = 0.0
        for start in range(0, total, eval_batch_size):
            end = min(total, start + eval_batch_size)
            batch_loss = loss(
                params,
                (jnp.asarray(state_x[start:end]), jnp.asarray(target_y[start:end])),
                0.0,
            )
            total_loss_sum += float(batch_loss)
        return total_loss_sum / total
    
    with tqdm(
        total=int(args.num_epochs),
        desc='train',
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    ) as progress:
        for iteration in range(args.num_epochs):
            rand_idx = np_rng.integers(0, len(data['x']), size=int(args.batch_size), endpoint=False)
            batch = (jnp.asarray(data['x'][rand_idx]), jnp.asarray(data[target_key][rand_idx]))
            opt_state, params, grad_norm = update_derivative(iteration, opt_state, batch, args.l2reg)
            grad_norm_value = float(grad_norm)
            if not np.isfinite(grad_norm_value):
                stop_reason = 'nan_grad_norm'
                stop_iteration = iteration
                break

            if iteration % param_check_every == 0 or iteration == args.num_epochs - 1:
                if not params_are_finite(params):
                    stop_reason = 'nan_params'
                    stop_iteration = iteration
                    break

            small_loss = loss(params, batch, 0.0)
            small_loss_value = float(small_loss)
            if not np.isfinite(small_loss_value):
                stop_reason = 'nan_batch_loss'
                stop_iteration = iteration
                break

            if latest_checkpoint_callback is not None and latest_checkpoint_every > 0 and (iteration + 1) % latest_checkpoint_every == 0:
                latest_checkpoint_callback(
                    iteration=int(iteration),
                    params=params,
                )

            new_small_loss = False
            if small_loss < best_small_loss:
                best_small_loss = small_loss
                new_small_loss = True

            if iteration % 100 == 0 or iteration == args.num_epochs - 1:
                progress.set_postfix(
                    step=int(iteration),
                    batch_loss=f"{small_loss_value:.4f}",
                    grad_norm=f"{grad_norm_value:.4f}",
                )

            eval_every = int(getattr(args, 'eval_every', 1000))
            warmup_eval_every = int(getattr(args, 'warmup_eval_every', 100))
            warmup_eval_until = int(getattr(args, 'warmup_eval_until', 1000))
            eval_on_small_loss_improve = bool(getattr(args, 'eval_on_small_loss_improve', True))
            final_eval_only = bool(getattr(args, 'final_eval_only', False))

            should_eval = False
            if final_eval_only:
                should_eval = bool(iteration == args.num_epochs - 1)
            elif eval_on_small_loss_improve and new_small_loss:
                should_eval = True
            if not final_eval_only:
                if iteration == 0 or iteration == args.num_epochs - 1:
                    should_eval = True
                elif warmup_eval_every > 0 and iteration < warmup_eval_until and iteration % warmup_eval_every == 0:
                    should_eval = True
                elif eval_every > 0 and iteration % eval_every == 0:
                    should_eval = True

            if should_eval:
                params = get_params(opt_state)
                eval_train_x = data['x'] if eval_train_idx is None else data['x'][eval_train_idx]
                eval_train_y = data[target_key] if eval_train_idx is None else data[target_key][eval_train_idx]
                eval_test_x = data['test_x'] if eval_test_idx is None else data['test_x'][eval_test_idx]
                eval_test_y = data['test_' + target_key] if eval_test_idx is None else data['test_' + target_key][eval_test_idx]

                train_loss = dataset_loss_batched(params, eval_train_x, eval_train_y)
                train_losses.append(train_loss)
                test_loss = dataset_loss_batched(params, eval_test_x, eval_test_y)
                test_losses.append(test_loss)

                progress.set_postfix(
                    step=int(iteration),
                    train_loss=f"{float(train_loss):.4f}",
                    test_loss=f"{float(test_loss):.4f}",
                )

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = params
                    best_iteration = iteration
                    if checkpoint_callback is not None:
                        checkpoint_callback(
                            iteration=int(iteration),
                            params=params,
                            test_loss=float(test_loss),
                        )

                if not np.isfinite(float(test_loss)):
                    stop_reason = 'nan_test_loss'
                    stop_iteration = iteration
                    break

                print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

            progress.update(1)

    params = get_params(opt_state)
    train_meta = {
        'completed': bool(stop_reason == 'completed'),
        'stop_reason': str(stop_reason),
        'stop_iteration': int(stop_iteration),
        'best_iteration': int(best_iteration),
        'best_loss': float(best_loss),
    }
    return params, train_losses, test_losses, best_loss

from matplotlib import pyplot as plt

# args = ObjectView(dict(
    # num_epochs=100, #40000
    # loss='l1',
    # l2reg=1e-6,
    # act='softplus',
    # hidden_dim=500,
    # output_dim=1,
    # dt=1e-1,
    # layers=2,
    # lr=1e-3*0.5,
    # lr2=1e-4*0.5,
    # model='gln',
    # n_updates=3,
    # batch_size=32,
# ))

def test_args(args):
    print('Running on', args.__dict__)
    rng = jax.random.PRNGKey(0)
    init_random_params, nn_forward_fn = extended_mlp(args)
    _, init_params = init_random_params(rng+1, (-1, 4))
    model = (nn_forward_fn, init_params)
    data = new_get_dataset(jax.random.PRNGKey(0), t_span=[0, dataset_size], fps=fps, samples=samples, test_split=0.9)

    result = train(args, model, data, rng+3)
    print(result[3], 'is the loss for', args.__dict__)

    if not jnp.isfinite(result[3]).sum():
        return {'status': 'fail', 'loss': float('inf')}
    return {'status': 'ok', 'loss': float(result[3])}

#test_args(args)
