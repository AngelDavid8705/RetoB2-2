# Reto B2-2 · Mantenimiento predictivo · Equipo AYJPS

Pipeline de datos y línea base de la Entrega 1 del proyecto de predicción de fallas con el conjunto Microsoft Azure Predictive Maintenance.

## Estructura

```
archive/                        CSV originales (no se suben al repositorio)
src/pipeline_datos.py           pipeline que genera train y val desde los CSV
Entrega1_RetoB2-2_AYJPS.ipynb   análisis exploratorio, etiquetas, fuga y línea base
data/processed/                 train.parquet y val.parquet generados
requirements.txt
```

## Cómo reproducir

1. Copiar los cinco CSV en la carpeta `archive`.
2. Instalar dependencias con `pip install -r requirements.txt`.
3. Generar los conjuntos con `python src/pipeline_datos.py archive`.
4. Abrir el notebook en Colab o Jupyter y ejecutar todas las celdas. En Colab el notebook busca `archive` en la sesión o en Google Drive y, si no la encuentra, pide subir el zip.

La semilla es 42. El conjunto de prueba (noviembre y diciembre de 2015) no se genera en esta entrega.

## Decisiones principales

* Etiqueta binaria, falla de cualquier componente en (t, t + 24].
* Etiqueta multiclase, componente de la falla más próxima en (t, t + 24], con desempate por menor número de componente.
* Entrenamiento con t del 1 de enero al 30 de agosto de 2015 y validación del 1 de septiembre al 30 de octubre, dejando 24 h de margen en cada frontera.
* Todas las variables usan solo datos con fecha menor o igual a t, lo que se verifica en el notebook con una prueba de causalidad.

## Referencias

* Microsoft. Predictive Maintenance Modelling Guide (Azure AI Gallery).
* Ke, G. et al. (2017). LightGBM, a highly efficient gradient boosting decision tree. NeurIPS.
