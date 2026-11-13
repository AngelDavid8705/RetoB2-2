# Reto B2-2 · Mantenimiento predictivo · Equipo AYJPS

Pipeline de datos y línea base de la Entrega 1 del proyecto de predicción de fallas con el conjunto Microsoft Azure Predictive Maintenance.

## Estructura

```
archive/                        CSV originales
src/pipeline_datos.py           pipeline que genera train y val desde los CSV
Entrega1_RetoB2-2_AYJPS.ipynb   análisis exploratorio, etiquetas, fuga y línea base
data/processed/                 train.parquet y val.parquet generados
requirements.txt
```

## Cómo reproducir

Abrir el notebook en Colab, subir la carpeta `archive` al panel de archivos de la sesión y ejecutar todas las celdas.
La semilla es 42. El conjunto de prueba (noviembre y diciembre de 2015) no se genera en esta entrega.

## Decisiones principales
* Etiqueta binaria, falla de cualquier componente en (t, t + 24].
* Etiqueta multiclase, componente de la falla más próxima en (t, t + 24], con desempate por menor número de componente.
* Entrenamiento con t del 1 de enero al 30 de agosto de 2015 y validación del 1 de septiembre al 30 de octubre, dejando 24 h de margen en cada frontera.
* Todas las variables usan solo datos con fecha menor o igual a t, lo que se verifica en el notebook con una prueba de causalidad.

## Por qué LightGBM obtiene PR-AUC de 1.000

La línea base con LightGBM llega a PR-AUC de 1.000 en validación. No es fuga de información, ya que las etiquetas respetan las fronteras de la partición y todas las variables usan solo datos hasta la hora t, lo que el notebook comprueba con código. Creemos que el resultado se explica por un patrón de la simulación de Microsoft.

* El sensor asociado a cada componente cambia unas 48 h antes de la falla.
* El 96.6 % de las fallas tiene un error registrado exactamente 24 h antes.

Como el conteo de errores que pide el punto 1.4 usa una ventana de 24 h, se enciende justo en las 24 horas positivas (el 100 % de las horas positivas tiene un error en esa ventana contra 8 % de las negativas). Junto con el salto del sensor marca exactamente la etiqueta, y LightGBM aprende esa combinación. Sin los conteos de error LightGBM baja a 0.923, y con ventana de 48 h queda en 0.997. Los detalles están al final de la sección 9 del notebook

## Referencias

* Microsoft. Predictive Maintenance Modelling Guide (Azure AI Gallery).
* Ke, G. et al. (2017). LightGBM, a highly efficient gradient boosting decision tree. NeurIPS.
