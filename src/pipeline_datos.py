"""Pipeline de datos del Reto B2-2 (Entrega 1).

Construye los conjuntos de entrenamiento y validacion a partir de los CSV
originales de Microsoft Azure Predictive Maintenance.
Todas las variables son causales, es decir, en la hora t solo usan datos con
fecha menor o igual a t. Las etiquetas miran la ventana futura (t, t + H].
"""
import os
import numpy as np
import pandas as pd

H = 24  # horizonte de prediccion en horas
SENSORES = ["volt", "rotate", "pressure", "vibration"]
COMPONENTES = ["comp1", "comp2", "comp3", "comp4"]
ERRORES = ["error1", "error2", "error3", "error4", "error5"]
CLASES = ["ninguno"] + COMPONENTES

# Particion temporal obligatoria. El fin es exclusivo y ya resta el margen de H horas,
# asi ninguna etiqueta depende de eventos posteriores al cierre del periodo.
PERIODOS = {
    "train": ("2015-01-01 00:00", "2015-08-31 00:00"),
    "val": ("2015-09-01 00:00", "2015-10-31 00:00"),
    # el conjunto de prueba (noviembre y diciembre de 2015) se construye hasta la semana 3
}


def cargar_datos(data_dir):
    """Lee los cinco CSV y convierte las fechas."""
    leer = lambda n: pd.read_csv(os.path.join(data_dir, f"PdM_{n}.csv"))
    tel, err, fal, man, maq = (leer(n) for n in ["telemetry", "errors", "failures", "maint", "machines"])
    for df in (tel, err, fal, man):
        df["datetime"] = pd.to_datetime(df["datetime"])
    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    return tel, err, fal, man, maq


def construir_etiquetas(tel, fal, h=H):
    """Etiqueta binaria y multiclase para cada (maquina, hora).

    y_bin = 1 si hay una falla de cualquier componente en (t, t + h].
    y_<comp> = 1 si ese componente falla en (t, t + h].
    y_multi toma la falla mas proxima. Si fallan varios componentes a la misma hora
    se asigna el de menor numero (pasa en 42 de 719 eventos).
    """
    base = tel[["machineID", "datetime"]].copy()
    filas = []
    for _, f in fal.iterrows():
        # horas t con t < fecha_falla <= t + h
        horas = pd.date_range(f["datetime"] - pd.Timedelta(hours=h), f["datetime"] - pd.Timedelta(hours=1), freq="h")
        filas.append(pd.DataFrame({"machineID": f["machineID"], "datetime": horas,
                                   "comp": f["failure"], "falla_dt": f["datetime"]}))
    pos = pd.concat(filas, ignore_index=True)

    multi = pos.pivot_table(index=["machineID", "datetime"], columns="comp", values="falla_dt",
                            aggfunc="size", fill_value=0).clip(upper=1)
    multi = multi.reindex(columns=COMPONENTES, fill_value=0).add_prefix("y_").reset_index()

    # clase principal, primero la falla mas proxima y despues el componente de menor numero
    pos["num"] = pos["comp"].str[-1].astype(int)
    prin = (pos.sort_values(["machineID", "datetime", "falla_dt", "num"])
               .drop_duplicates(["machineID", "datetime"])[["machineID", "datetime", "num"]])

    base = base.merge(multi, on=["machineID", "datetime"], how="left").merge(prin, on=["machineID", "datetime"], how="left")
    ycols = [f"y_{c}" for c in COMPONENTES]
    base[ycols] = base[ycols].fillna(0).astype(int)
    base["y_multi"] = base["num"].fillna(0).astype(int)  # 0 ninguno, 1..4 comp1..comp4
    base["y_bin"] = (base["y_multi"] > 0).astype(int)
    return base.drop(columns="num")


def construir_variables(tel, err, man, maq, ventana_errores=24):
    """Variables de la linea base, todas calculadas con datos hasta t inclusive.

    ventana_errores permite cambiar la ventana del conteo de errores (24 h por defecto).
    """
    X = tel.copy()
    g = X.groupby("machineID")
    for s in SENSORES:
        for w in (3, 24):
            X[f"{s}_media_{w}h"] = g[s].transform(lambda v: v.rolling(w, min_periods=1).mean())
            X[f"{s}_std_{w}h"] = g[s].transform(lambda v: v.rolling(w, min_periods=2).std())

    # conteo de cada error en las ultimas w horas, ventana (t - w, t]
    w = ventana_errores
    e = err.assign(n=1).pivot_table(index=["machineID", "datetime"], columns="errorID", values="n",
                                    aggfunc="sum", fill_value=0).reindex(columns=ERRORES, fill_value=0).reset_index()
    X = X.merge(e, on=["machineID", "datetime"], how="left")
    X[ERRORES] = X[ERRORES].fillna(0)
    g = X.groupby("machineID")
    for c in ERRORES:
        X[f"{c}_cuenta_{w}h"] = g[c].transform(lambda v: v.rolling(w, min_periods=1).sum())
    X = X.drop(columns=ERRORES)

    # horas desde el ultimo reemplazo de cada componente (registros de mantenimiento con fecha <= t)
    for c in COMPONENTES:
        r = man.loc[man["comp"] == c, ["machineID", "datetime"]].rename(columns={"datetime": "ult"}).sort_values("ult")
        X = pd.merge_asof(X.sort_values("datetime"), r, left_on="datetime", right_on="ult",
                          by="machineID", direction="backward")
        X[f"horas_desde_{c}"] = (X["datetime"] - X["ult"]).dt.total_seconds() / 3600
        X = X.drop(columns="ult")
    X = X.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    # metadatos estaticos
    X = X.merge(maq, on="machineID", how="left")
    X = pd.concat([X, pd.get_dummies(X["model"], prefix="modelo", dtype=int)], axis=1).drop(columns="model")
    return X


def particionar(df):
    """Devuelve un diccionario con los conjuntos segun la particion temporal."""
    return {k: df[(df["datetime"] >= a) & (df["datetime"] < b)].reset_index(drop=True)
            for k, (a, b) in PERIODOS.items()}


def generar_conjuntos(data_dir, out_dir="data/processed"):
    """Ejecuta el pipeline completo y guarda train y val en parquet."""
    tel, err, fal, man, maq = cargar_datos(data_dir)
    X = construir_variables(tel, err, man, maq)
    Y = construir_etiquetas(tel, fal)
    df = X.merge(Y, on=["machineID", "datetime"], how="left")
    conj = particionar(df)
    os.makedirs(out_dir, exist_ok=True)
    for k, d in conj.items():
        d.to_parquet(os.path.join(out_dir, f"{k}.parquet"), index=False)
    return conj


if __name__ == "__main__":
    import sys
    c = generar_conjuntos(sys.argv[1] if len(sys.argv) > 1 else "archive")
    for k, d in c.items():
        print(k, d.shape, "positivos", round(d["y_bin"].mean() * 100, 2), "%")
