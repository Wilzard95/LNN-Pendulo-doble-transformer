# [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630)
Miles Cranmer, Sam Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel, Shirley Ho

![overall-idea.png](static/overall-idea.png)

* [Paper](https://arxiv.org/abs/2003.04630)
* [Blog](https://greydanus.github.io/2020/03/10/lagrangian-nns/)
* [Self-Contained Tutorial](notebooks/LNN_Tutorial.ipynb)
* [Paper example notebook: double pendulum](notebooks/DoublePendulum.ipynb)
* [Paper example notebook: special relativity](notebooks/SpecialRelativity.ipynb)
* [Paper example notebook: wave equation](notebooks/WaveEquation.ipynb)

> [!WARNING]
> This project was developed with JAX 0.1.55 (2020). For compatibility, use pixi to install the original environment ([see guide](#quick-start)). Alternatively, adapt the [core equation snippets](#core-equation-of-motion) below for modern JAX.

## Summary

In this project we propose Lagrangian Neural Networks (LNNs), which can parameterize arbitrary Lagrangians using neural networks. In contrast to Hamiltonian Neural Networks, these models do not require canonical coordinates and perform well in situations where generalized momentum is difficult to compute (e.g., the double pendulum). This is particularly appealing for use with a learned latent representation, a case where HNNs struggle. Unlike [previous work on learning Lagrangians](https://arxiv.org/pdf/1907.04490.pdf), LNNs are fully general and extend to non-holonomic systems such as the 1D wave equation.

|	| Neural Networks  | [Neural ODEs](https://arxiv.org/abs/1806.07366) | [HNN](https://arxiv.org/abs/1906.01563)  | [DLN (ICLR'19)](https://arxiv.org/abs/1907.04490) | LNN (this work) |
| ------------- |:------------:| :------------:| :------------:| :------------:| :------------:|
| Learns dynamics | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| Learns continuous-time dynamics | | ✔️ | ✔️ | ✔️ | ✔️ |
| Learns exact conservation laws | | | ✔️ | ✔️ | ✔️ |
| Learns from arbitrary coordinates| ✔️ | ✔️ || ✔️ | ✔️ |
| Learns arbitrary Lagrangians | | |  | | ✔️ |


## Core Equation of Motion

The key innovation of LNNs is the automatic derivation of equations of motion from learned Lagrangians. The core equation of motion is **version-independent** and can be adapted to any JAX version:

```python
import jax
import jax.numpy as jnp

def lagrangian_eom(lagrangian, state, t=None):
    """Compute Euler-Lagrange equation of motion.

    Args:
        lagrangian: A function L(q, q_dot) representing the Lagrangian
        state: Concatenated position q and velocity q_dot

    Returns:
        Concatenated velocity and acceleration
    """
    q, q_dot = jnp.split(state, 2)

    # Euler-Lagrange equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0
    # Rearranged to solve for acceleration q_ddot
    q_ddot = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_dot))
              @ (jax.grad(lagrangian, 0)(q, q_dot)
                 - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_dot) @ q_dot))

    return jnp.concatenate([q_dot, q_ddot])


# Custom initialization for MLPs learning Lagrangians
def custom_init(layer_sizes, seed=0):
    """Initialize MLP to learn Lagrangians more effectively."""
    # See lnn/core.py for full implementation
    # Uses optimized scale for each layer
```

This equation of motion works with any JAX version. The rest of the codebase is maintained with JAX 0.1.55 for reproducibility of the paper's environment.


## Installation

This project requires specific versions of JAX and other dependencies from early 2020 to ensure reproducibility. We use [pixi](https://pixi.sh) for reproducible environment management.

### Quick Start

1. **Install pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```
   Or via Homebrew:
   ```bash
   brew install pixi
   ```

2. **Clone this repository and install dependencies**:
   ```bash
   git clone https://github.com/MilesCranmer/lagrangian_nns.git
   cd lagrangian_nns
   pixi install
   ```

3. **Run the notebooks**:
   ```bash
   pixi run jupyter notebook
   ```

### Dependencies

The environment includes:
 * Python 3.7
 * JAX 0.1.55 & jaxlib 0.1.37 (January 2020 versions)
 * NumPy 1.19.5
 * Matplotlib 3.1.2 (visualization)
 * MoviePy 1.0.0 (visualization)
 * celluloid 0.2.0 (visualization)

All dependencies are pinned to versions from early 2020 to ensure the code behaves exactly as it did when the paper was published.
