# Reproduciendo LNN en pendulo doble (Windows)

Proyecto completo en Python + PyTorch para entrenar una Lagrangian Neural Network (LNN) con datos de pendulo doble, evaluar en test y hacer rollout.

## Rutas por defecto

- Proyecto: `C:\Users\Nitro\Documents\Documentos William\Modelos\Reproduciendo LNN`
- Dataset: `C:\Users\Nitro\Documents\Documentos William\Creando_mi_IA\Probando_modelos\Pendulos\dataset_pendulo_doble_500`

## 1) Crear entorno e instalar dependencias

```powershell
cd "C:\Users\Nitro\Documents\Documentos William\Modelos\Reproduciendo LNN"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Sanity check del dataset

```powershell
python sanity_check_data.py
```

Salida esperada:
- cantidad de archivos leidos
- chequeo de NaN
- rangos de variables
- resumen de `dt`
- figura ejemplo en `plots/`

## 3) Entrenamiento LNN

```powershell
python train_lnn.py
```

Comando ejemplo con ajustes:

```powershell
python train_lnn.py --epochs 200 --batch_size 2048 --lr 1e-3 --lambda_damp 1e-4 --split_by_trajectory true
python train_lnn.py --device cuda --epochs 200 --batch_size 2048 --lr 1e-3 --lambda_damp 1e-4 --split_by_trajectory true
```

Durante entrenamiento imprime por epoca:
- `epoch`
- `train_loss`
- `val_loss`
- `lr`

Nota:
- por defecto se usa `--device cuda` (si CUDA no esta disponible, el script falla con error explicito)

Archivos de salida:
- `checkpoints/model_best.pth`
- `model_final.pth`
- `metrics.csv`
- `run_config.json`
- `results/split_info.json`

## 4) Evaluacion en test

```powershell
python eval_lnn.py
```

Comando ejemplo:

```powershell
python eval_lnn.py --model_path "C:\Users\Nitro\Documents\Documentos William\Modelos\Reproduciendo LNN\checkpoints\model_best.pth"
```

Genera:
- `results/eval_metrics.json`
- graficas en `plots/`:
  - scatter `qddot_true` vs `qddot_pred`
  - histogramas de error por componente
  - curva de perdidas train/val

## 5) Rollout dinamico

Usa RK4 por defecto e infiere `dt` del archivo real.

```powershell
python rollout_lnn.py
```

Ejemplos:

```powershell
python rollout_lnn.py --data_file sim_data_123.txt --horizon_s 5.0 --integrator rk4
python rollout_lnn.py --data_file sim_data_123.txt --n_steps 1500 --integrator euler
```

Genera:
- `plots/rollout_<archivo>.png`
- `results/rollout_<archivo>.csv`

## Estructura

```text
Reproduciendo LNN/
  README.md
  requirements.txt
  train_lnn.py
  eval_lnn.py
  rollout_lnn.py
  sanity_check_data.py
  lnn/
    __init__.py
    data.py
    model.py
    dynamics.py
    integrators.py
    metrics.py
    plotting.py
    utils.py
```
