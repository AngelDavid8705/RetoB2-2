# Reto B2-2 · Mantenimiento predictivo · Equipo AYJPS

Pipeline de datos, línea base (Entrega 1), modelos secuenciales de deep learning (Entrega 2) y modelo avanzado con política de mantenimiento (Entrega 3) del proyecto de predicción de fallas con el conjunto Microsoft Azure Predictive Maintenance.

## Estructura

```
archive/                        CSV originales 
src/pipeline_datos.py           pipeline que genera train y val desde los CSV
src/modelos_dl.py               tensor de entrada, GRU, TCN, CNN-LSTM, Transformer, métricas, calibración y políticas
Entrega1_RetoB2-2_AYJPS.ipynb   análisis exploratorio, etiquetas, fuga y línea base
Entrega2_RetoB2-2_AYJPS.ipynb   modelos de deep learning y comparación con la línea base
Entrega3_RetoB2-2_AYJPS.ipynb   Transformer, ensamble, calibración, interpretabilidad y política de mantenimiento
resultados/                     tablas y figuras de la entrega 1
resultados_e2/                  figuras de la entrega 2
resultados_e3/                  figuras de la entrega 3
data/processed/                 train.parquet y val.parquet generados
requirements.txt
```

## Cómo reproducir

1. Copiar los cinco CSV en la carpeta `archive`.
2. Instalar dependencias con `pip install -r requirements.txt`.
3. Generar los conjuntos con `python src/pipeline_datos.py archive`.
4. Abrir el notebook en Colab, subir la carpeta `archive` al panel de archivos de la sesión y ejecutar todas las celdas.

La semilla es 42.
## Decisiones principales

* Etiqueta binaria, falla de cualquier componente en (t, t + 24].
* Etiqueta multiclase, componente de la falla más próxima en (t, t + 24], con desempate por menor número de componente.
* Entrenamiento con t del 1 de enero al 30 de agosto de 2015 y validación del 1 de septiembre al 30 de octubre, dejando 24 h de margen en cada frontera.
* Todas las variables usan solo datos con fecha menor o igual a t, lo que se verifica en el notebook con una prueba de causalidad.

## Por qué LightGBM obtiene PR-AUC de 1.000

La línea base con LightGBM llega a PR-AUC de 1.000 en validación. No es fuga de información, ya que las etiquetas respetan las fronteras de la partición y todas las variables usan solo datos hasta la hora t, lo que el notebook comprueba con código. El resultado se explica por un patrón de la simulación de Microsoft.

* El sensor asociado a cada componente cambia unas 48 h antes de la falla.
* El 96.6 % de las fallas tiene un error registrado exactamente 24 h antes.

Como el conteo de errores que pide el punto 1.4 usa una ventana de 24 h, se enciende justo en las 24 horas positivas (el 100 % de las horas positivas tiene un error en esa ventana contra 8 % de las negativas). Junto con el salto del sensor marca exactamente la etiqueta, y LightGBM aprende esa combinación. Sin los conteos de error LightGBM baja a 0.923, y con ventana de 48 h queda en 0.997. En una planta real los errores de aviso no llegarían siempre con la misma anticipación, así que este resultado no sería alcanzable.

## Entrega 2, modelos secuenciales

* **Sin errores como entrada.** Por el patrón descrito arriba, las redes no usan los indicadores de error y la línea base de comparación es LightGBM sin conteos de error (PR-AUC 0.923). Un experimento aparte agrega los canales de error para mostrar su efecto.
* **Tensor de entrada.** Ventana de L horas con los cuatro sensores y las horas desde el último reemplazo de cada componente (escala log), normalización global con estadísticas de entrenamiento, más embedding del modelo de máquina y antigüedad.
* **Arquitecturas.** GRU, TCN y CNN-LSTM con salida binaria y multiclase y pérdida L_bin + λ · L_multi.
* **Experimentos.** L de 24, 72 y 168 h, pérdida ponderada contra focal loss y submuestreo de horas sanas, y Optuna con 15 ensayos de 4 épocas de 30 000 ventanas.
* **Resultado.** La mejor red (TCN, L = 168, pérdida ponderada) llega a PR-AUC de 0.772 y F1 macro de 0.750 contra 0.923 y 0.856 de LightGBM, con 24.4 h de anticipación media contra 23.7 h. La búsqueda con Optuna no mejoró al modelo (0.733). Con canales de error la misma red llega a 0.997.

Los resultados se obtuvieron en Colab con GPU, con 200 000 ventanas por época y hasta 30 épocas.

## Entrega 3, modelo avanzado y política de mantenimiento

* **Modelo avanzado.** Codificador Transformer sobre parches de 4 horas con codificación posicional y un token de máquina, sin errores como entrada. Un modelo llega a PR-AUC de 0.680 y el ensamble de 5 semillas a 0.758 en validación.
* **Incertidumbre y calibración.** Ensamble de 5 modelos y temperature scaling (T = 1.37). La discrepancia entre modelos es mucho mayor en las fallas no detectadas.
* **Interpretabilidad.** Gradientes integrados por canal y hora. Cada componente se explica por su sensor (comp1 voltaje, comp2 rotación, comp3 presión, comp4 vibración) en las últimas 36 horas.
* **Política.** El umbral y el modelo se eligen minimizando el costo en validación con los costos de la Tabla 3. Ganó LightGBM sin errores. En prueba, evaluada una sola vez, evita 109 de 110 fallas con 4.4 falsas alarmas por semana y 19.5 h de anticipación, con un costo anual estimado de 2.2 millones contra 10.0 millones de operar hasta la falla y 10.1 millones del reemplazo por calendario.
* **Sensibilidad.** La política del modelo es la más barata para toda razón entre costo de falla y de falsa alarma de 10 a 100, y empata con operar hasta la falla cuando la falla cuesta lo mismo que una intervención.
* **Pruebas de control.** Con las etiquetas de entrenamiento revueltas el PR-AUC de LightGBM en prueba cae a 0.025, igual al azar, lo que descarta una fuga de información. Los sensores solos llegan a 0.25 y las horas desde el reemplazo solas a 0.36, y juntos a 0.93.

Los resultados de la entrega 3 se obtuvieron en CPU con 80 000 ventanas por época y hasta 20 épocas para cada Transformer. En GPU las cifras pueden variar ligeramente.

## Referencias

* Microsoft. Predictive Maintenance Modelling Guide (Azure AI Gallery).
* Ke, G. et al. (2017). LightGBM, a highly efficient gradient boosting decision tree. NeurIPS.
* Bai, S., Kolter, J. Z. y Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv 1803.01271.
* Lin, T. Y. et al. (2017). Focal loss for dense object detection. ICCV.
* Akiba, T. et al. (2019). Optuna, a next-generation hyperparameter optimization framework. KDD.
* Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.
* Lakshminarayanan, B., Pritzel, A. y Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS.
* Guo, C. et al. (2017). On calibration of modern neural networks. ICML.
* Sundararajan, M., Taly, A. y Yan, Q. (2017). Axiomatic attribution for deep networks. ICML.
