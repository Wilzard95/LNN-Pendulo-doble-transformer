# Official JAX Route

Esta carpeta ya contiene el repo oficial clonado en:

- `official_lagrangian_nns`

## Lo realmente "exacto"

La reproduccion exacta de la publicacion requiere el entorno que el propio repo declara:

- Python 3.7
- JAX 0.1.55
- jaxlib 0.1.37
- pixi
- plataforma `linux-64` u `osx-64`

En este Windows nativo no se puede ejecutar ese entorno tal cual:

- `pixi` no esta instalado
- el `pyproject.toml` oficial no declara soporte `win-64`
- WSL no esta instalado en esta maquina

## Lo que si dejamos listo aqui

Se preparo una ruta de compatibilidad minima para correr el codigo oficial en CPU con el JAX moderno que ya tienes:

- runner: `train_official_double_pendulum_cpu.py`
- repo oficial: `official_lagrangian_nns`

Los unicos cambios hechos al repo oficial fueron de compatibilidad:

- imports `jax.experimental.{stax,optimizers}` -> compat moderna
- `moviepy` se volvio opcional para que el entrenamiento no dependa del stack de video
- se removio `mxsteps` del trayecto de `HyperparameterSearch`, porque `odeint` moderno ya no lo acepta

La logica del experimento oficial no se cambio.

## Entrenamiento oficial en CPU

Desde `Otrointento`:

```powershell
& "..\.venv\Scripts\python.exe" "train_official_double_pendulum_cpu.py" --out_dir "experiments\official_jax_cpu"
```

Con reintentos automaticos por `seed`:

```powershell
& "..\.venv\Scripts\python.exe" "train_official_double_pendulum_cpu.py" --out_dir "experiments\official_jax_cpu" --max_attempts 5
```

Smoke test rapido:

```powershell
& "..\.venv\Scripts\python.exe" "train_official_double_pendulum_cpu.py" --out_dir "experiments\official_jax_cpu_smoke" --smoke
```

## Que significa esta ruta

Esto no es una reproduccion bit-a-bit del paper.
Si es la ruta mas fiel que hoy podemos correr en esta maquina sin WSL ni Linux:

- mismo repo oficial
- misma fisica
- mismo dataset sintetico
- misma loss
- mismo integrador RK4 del repo
- mismos hiperparametros "best" del notebook

Si luego quieres la reproduccion estricta de publicacion, el siguiente paso es instalar WSL y correr el repo oficial dentro de Linux con el entorno pinneado.

## Nuevos artefactos del runner

Cada intento queda aislado en:

- `experiments/.../attempts/attempt_XX_seed_YYYY`

Y arriba del `out_dir` se promociona automaticamente el mejor intento:

- `checkpoints/model_best.pkl`
- `model_final.pkl`
- `results/training.log`
- `results/loss_history.csv`
- `plots/loss_curve.png`
- `results/attempts_summary.json`

Si un intento cae en `NaN`, el runner ahora deja:

- `stop_reason`
- `stop_iteration`
- `best_iteration`

en el resumen del intento y en el resumen global.
