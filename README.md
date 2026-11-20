# Reto B2-2 · Mantenimiento predictivo · Equipo AYJPS

Pipeline de datos, línea base (Entrega 1) y modelos secuenciales de deep learning (Entrega 2) del proyecto de predicción de fallas con el conjunto Microsoft Azure Predictive Maintenance.

## Estructura

```
archive/                        CSV originales (no se suben al repositorio)
src/pipeline_datos.py           pipeline que genera train y val desde los CSV
src/modelos_dl.py               tensor de entrada, GRU, TCN, CNN-LSTM, entrenamiento y métricas
Entrega1_RetoB2-2_AYJPS.ipynb   análisis exploratorio, etiquetas, fuga y línea base
Entrega2_RetoB2-2_AYJPS.ipynb   modelos de deep learning y comparación con la línea base
resultados/                     tablas y figuras de la entrega 1
resultados_e2/                  tablas y figuras de la entrega 2
data/processed/                 train.parquet y val.parquet generados
requirements.txt
```

## Cómo reproducir

1. Copiar los cinco CSV en la carpeta `archive`.
2. Instalar dependencias con `pip install -r requirements.txt`.
3. Generar los conjuntos con `python src/pipeline_datos.py archive`.
4. Abrir el notebook en Colab, subir la carpeta `archive` al panel de archivos de la sesión y ejecutar todas las celdas. Para la entrega 2 se recomienda activar la GPU. Con `RAPIDO = True` el notebook corre una versión corta de prueba.

La semilla es 42. El conjunto de prueba (noviembre y diciembre de 2015) no se genera en esta entrega.

## Decisiones principales

* Etiqueta binaria, falla de cualquier componente en (t, t + 24].
* Etiqueta multiclase, componente de la falla más próxima en (t, t + 24], con desempate por menor número de componente.
* Entrenamiento con t del 1 de enero al 30 de agosto de 2015 y validación del 1 de septiembre al 30 de octubre, dejando 24 h de margen en cada frontera.
* Todas las variables usan solo datos con fecha menor o igual a t, lo que se verifica en el notebook con una prueba de causalidad.

## Por qué LightGBM obtiene PR-AUC de 1.000

La línea base con LightGBM llega a PR-AUC de 1.000 en validación. No es fuga de información, ya que las etiquetas respetan las fronteras de la partición y todas las variables usan solo datos hasta la hora t, lo que el notebook comprueba con código. El resultado se explica por un patrón de la simulación de Microsoft.

* El sensor asociado a cada componente cambia unas 48 h antes de la falla.
* El 96.6 % de las fallas tiene un error registrado exactamente 24 h antes.

Como el conteo de errores que pide el punto 1.4 usa una ventana de 24 h, se enciende justo en las 24 horas positivas (el 100 % de las horas positivas tiene un error en esa ventana contra 8 % de las negativas). Junto con el salto del sensor marca exactamente la etiqueta, y LightGBM aprende esa combinación. Sin los conteos de error LightGBM baja a 0.923, y con ventana de 48 h queda en 0.997. Los detalles están al final de la sección 9 del notebook. En una planta real los errores de aviso no llegarían siempre con la misma anticipación, así que este resultado no sería alcanzable.

## Entrega 2, modelos secuenciales

* **Sin errores como entrada.** Por el patrón descrito arriba, las redes no usan los indicadores de error y la línea base de comparación es LightGBM sin conteos de error (PR-AUC 0.923). Un experimento aparte agrega los canales de error para mostrar su efecto.
* **Tensor de entrada.** Ventana de L horas con los cuatro sensores y las horas desde el último reemplazo de cada componente (escala log), normalización global con estadísticas de entrenamiento, más embedding del modelo de máquina y antigüedad.
* **Arquitecturas.** GRU, TCN y CNN-LSTM con salida binaria y multiclase y pérdida L_bin + λ · L_multi.
* **Experimentos.** L de 24, 72 y 168 h, pérdida ponderada contra focal loss y submuestreo de horas sanas, y Optuna con 15 ensayos de 4 épocas.
* **Resultado.** El mejor modelo (CNN-LSTM, L = 168, submuestreo, ajustado) llega a PR-AUC de 0.757 y F1 macro de 0.728 contra 0.923 y 0.856 de LightGBM, con 26.5 h de anticipación media contra 23.7 h. Con canales de error la misma red llega a 0.996.

Los experimentos completos tardan alrededor de una hora en CPU y bastante menos con GPU.

## Referencias

* Microsoft. Predictive Maintenance Modelling Guide (Azure AI Gallery).
* Ke, G. et al. (2017). LightGBM, a highly efficient gradient boosting decision tree. NeurIPS.
* Bai, S., Kolter, J. Z. y Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv 1803.01271.
* Lin, T. Y. et al. (2017). Focal loss for dense object detection. ICCV.
* Akiba, T. et al. (2019). Optuna, a next-generation hyperparameter optimization framework. KDD.
