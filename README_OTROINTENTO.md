## Otrointento

Este directorio es el nuevo workspace limpio para intentar reproducir el paper de LNN.

Que significa "video libre":
- Es un rollout generado solo por el modelo a partir de un estado inicial elegido.
- No esta atado a una trayectoria real del dataset durante ese video.
- Sirve como inspeccion cualitativa, pero no es la prueba principal de reproduccion.

Por que se abrio este workspace:
- El run previo no era fiel al setup del paper.
- Se estaba usando un preset que en la practica entreno con una sola trayectoria.
- El pipeline mezclaba objetivos, integradores y datos distintos a los del repo original.

Diferencias detectadas contra el setup oficial:
- El preset `paper` usa una sola trayectoria por defecto y split ordenado.
- El run previo entreno `continuous -> xdot` en full batch.
- El repo original usa una ruta mas cercana a `delta` + minibatch + integracion RK4.
- El dataset local tiene fisica distinta a la del ejemplo oficial del repo.
- En el notebook oficial, el dataset usa `times = linspace(0, 50, 500)` y el mejor hiperparametro reportado mantiene `dt ~= 0.0961`; esa diferencia no es un bug del port.

Objetivo de este workspace:
1. Conservar lo util del codigo actual.
2. Alinear datos, objetivo, entrenamiento e integracion con el repo/paper original.
3. Validar primero rollout corto y energia.
4. Dejar el video libre solo como chequeo secundario.

Primeros pasos sugeridos:
1. Crear un modo `repo_faithful` con generacion de datos compatible con el repo original.
2. Separar claramente evaluacion `true vs pred` de video libre.
3. Entrenar smoke tests pequenos en este directorio.

Scripts nuevos en este workspace:
- `train_lnn_repo_faithful.py`: entrenamiento sintetico mas cercano al repo original.
- `eval_repo_faithful.py`: evaluacion one-step y rollout corto sobre trayectorias sinteticas.
- `lnn/repo_faithful_data.py`: generacion deterministica del dataset sintetico usado por esos scripts.
