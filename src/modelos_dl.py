"""Modelos secuenciales de deep learning del Reto B2-2 (Entregas 2 y 3).

Construye el tensor de entrada por ventanas, define las tres arquitecturas
(GRU, TCN y CNN-LSTM) con dos salidas y contiene el entrenamiento y la evaluacion.
Usa las funciones de src/pipeline_datos.py para cargar datos y etiquetas.
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             f1_score, brier_score_loss)

SENSORES = ["volt", "rotate", "pressure", "vibration"]
COMPONENTES = ["comp1", "comp2", "comp3", "comp4"]
ERRORES = ["error1", "error2", "error3", "error4", "error5"]
L_MAX = 168
INICIO_TRAIN, FIN_TRAIN = pd.Timestamp("2015-01-01"), pd.Timestamp("2015-08-31")
INICIO_VAL, FIN_VAL = pd.Timestamp("2015-09-01"), pd.Timestamp("2015-10-31")
INICIO_TEST, FIN_TEST = pd.Timestamp("2015-11-01"), pd.Timestamp("2015-12-31")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# tensor de entrada

def construir_tensor(tel, err, man, maq, etiquetas, con_errores=False):
    """Devuelve un diccionario con el tensor dinamico, los estaticos, las etiquetas y los indices.

    Canales dinamicos por hora, los cuatro sensores y las horas desde el ultimo reemplazo
    de cada componente (en escala log). Con con_errores=True se agregan cinco indicadores
    horarios de error, solo para el experimento de comparacion.
    La normalizacion es global y usa solo horas del periodo de entrenamiento.
    """
    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    maquinas = np.sort(tel.machineID.unique())
    horas = np.sort(tel.datetime.unique())
    M, T = len(maquinas), len(horas)
    base = tel[["machineID", "datetime"]]

    dyn = {s: tel[s].values for s in SENSORES}
    # horas desde el ultimo reemplazo de cada componente, con registros de fecha <= t
    for c in COMPONENTES:
        r = man.loc[man.comp == c, ["machineID", "datetime"]].rename(columns={"datetime": "ult"}).sort_values("ult")
        q = pd.merge_asof(base.sort_values("datetime"), r, left_on="datetime", right_on="ult",
                          by="machineID", direction="backward").sort_values(["machineID", "datetime"])
        dyn[f"log_horas_{c}"] = np.log1p((q.datetime - q.ult).dt.total_seconds().values / 3600)
    if con_errores:
        e = err.assign(n=1).pivot_table(index=["machineID", "datetime"], columns="errorID", values="n",
                                        aggfunc="sum", fill_value=0).reindex(columns=ERRORES, fill_value=0)
        e = base.merge(e.reset_index(), on=["machineID", "datetime"], how="left").fillna(0)
        for c in ERRORES:
            dyn[f"ind_{c}"] = e[c].values
    canales = list(dyn)
    X = np.stack([dyn[c] for c in canales], axis=1).astype(np.float32).reshape(M, T, len(canales))

    # normalizacion global con estadisticas solo de entrenamiento (los indicadores de error no se escalan)
    en_train = (horas >= np.datetime64(INICIO_TRAIN)) & (horas < np.datetime64(FIN_TRAIN))
    escalar = [i for i, c in enumerate(canales) if not c.startswith("ind_")]
    media = X[:, en_train][..., escalar].reshape(-1, len(escalar)).mean(0)
    desv = X[:, en_train][..., escalar].reshape(-1, len(escalar)).std(0)
    X[..., escalar] = (X[..., escalar] - media) / desv
    # relleno al inicio repitiendo la primera hora para ventanas sin historia completa
    X = np.concatenate([np.repeat(X[:, :1], L_MAX, axis=1), X], axis=1)

    maq = maq.set_index("machineID").loc[maquinas]
    modelo = maq["model"].str[-1].astype(int).values - 1
    edad = ((maq["age"] - maq["age"].mean()) / maq["age"].std()).values.astype(np.float32)

    Y = etiquetas.set_index(["machineID", "datetime"]).loc[pd.MultiIndex.from_frame(base)]
    y_bin = Y["y_bin"].values.reshape(M, T).astype(np.float32)
    y_multi = Y["y_multi"].values.reshape(M, T).astype(np.int64)

    def indices(a, b):
        tt = np.where((horas >= np.datetime64(a)) & (horas < np.datetime64(b)))[0]
        mm, ti = np.meshgrid(np.arange(M), tt, indexing="ij")
        return np.stack([mm.ravel(), ti.ravel()], 1)

    return {"X": torch.tensor(X), "modelo": torch.tensor(modelo), "edad": torch.tensor(edad),
            "y_bin": torch.tensor(y_bin), "y_multi": torch.tensor(y_multi), "canales": canales,
            "horas": horas, "maquinas": maquinas,
            "idx_train": indices(INICIO_TRAIN, FIN_TRAIN), "idx_val": indices(INICIO_VAL, FIN_VAL),
            "idx_test": indices(INICIO_TEST, FIN_TEST),
            "media": media, "desv": desv}


def lote(D, idx, L):
    """Arma un lote de ventanas (B, L, C) para los pares (maquina, hora) de idx."""
    idx = torch.as_tensor(idx)
    m, t = idx[:, 0], idx[:, 1] + L_MAX
    pasos = t[:, None] + torch.arange(-L + 1, 1)[None, :]
    x = D["X"][m[:, None], pasos]
    return (x.to(DEVICE), D["modelo"][m].to(DEVICE), D["edad"][m].to(DEVICE),
            D["y_bin"][idx[:, 0], idx[:, 1]].to(DEVICE), D["y_multi"][idx[:, 0], idx[:, 1]].to(DEVICE))


# arquitecturas

class Cabezas(nn.Module):
    """Concatena el resumen de la secuencia con los estaticos y produce las dos salidas."""
    def __init__(self, d_in, dropout):
        super().__init__()
        self.emb = nn.Embedding(4, 3)
        self.mlp = nn.Sequential(nn.Linear(d_in + 4, 64), nn.ReLU(), nn.Dropout(dropout))
        self.bin = nn.Linear(64, 1)
        self.multi = nn.Linear(64, 5)

    def forward(self, h, modelo, edad):
        z = self.mlp(torch.cat([h, self.emb(modelo), edad[:, None]], 1))
        return self.bin(z).squeeze(1), self.multi(z)


class GRUNet(nn.Module):
    def __init__(self, c_in, hidden=64, capas=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(c_in, hidden, num_layers=capas, batch_first=True,
                          dropout=dropout if capas > 1 else 0)
        self.cab = Cabezas(hidden, dropout)

    def forward(self, x, modelo, edad):
        _, h = self.rnn(x)
        return self.cab(h[-1], modelo, edad)


class BloqueCausal(nn.Module):
    def __init__(self, c_in, c_out, k, dil, dropout):
        super().__init__()
        self.pad = (k - 1) * dil
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=dil)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=dil)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.c2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.res(x))


class TCN(nn.Module):
    """Convoluciones causales dilatadas, el campo receptivo cubre la ventana L."""
    def __init__(self, c_in, hidden=32, L=72, k=3, dropout=0.2):
        super().__init__()
        bloques, dil, c, campo = [], 1, c_in, 1
        while campo < L:
            bloques.append(BloqueCausal(c, hidden, k, dil, dropout))
            campo += 2 * (k - 1) * dil
            c, dil = hidden, dil * 2
        self.red = nn.Sequential(*bloques)
        self.cab = Cabezas(hidden, dropout)

    def forward(self, x, modelo, edad):
        y = self.red(x.transpose(1, 2))
        return self.cab(y[:, :, -1], modelo, edad)


class CNNLSTM(nn.Module):
    """Dos convoluciones con submuestreo temporal seguidas de una LSTM."""
    def __init__(self, c_in, hidden=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c_in, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.rnn = nn.LSTM(32, hidden, batch_first=True)
        self.cab = Cabezas(hidden, dropout)

    def forward(self, x, modelo, edad):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.rnn(y)
        return self.cab(h[-1], modelo, edad)


class TransformerNet(nn.Module):
    """Codificador Transformer sobre parches de horas con un token de maquina al inicio.

    Cada parche agrupa `parche` horas consecutivas y se proyecta a dimension d. Se suma una
    codificacion posicional sinusoidal y se antepone un token de maquina construido con el
    embedding del modelo y la antiguedad. La salida de ese token resume la ventana.
    """
    def __init__(self, c_in, L=168, d=64, cabezas=4, capas=2, parche=4, dropout=0.1):
        super().__init__()
        self.parche = parche
        self.n_tok = -(-L // parche)
        self.proy = nn.Linear(c_in * parche, d)
        pos = torch.arange(self.n_tok)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe = torch.zeros(self.n_tok, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.tok_modelo = nn.Embedding(4, d)
        self.tok_edad = nn.Linear(1, d)
        capa = nn.TransformerEncoderLayer(d, cabezas, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(capa, capas)
        self.norm = nn.LayerNorm(d)
        self.cab = Cabezas(d, dropout)

    def forward(self, x, modelo, edad):
        B, L, C = x.shape
        resto = (-L) % self.parche
        if resto:
            x = torch.cat([x[:, :1].expand(B, resto, C), x], 1)
        z = self.proy(x.reshape(B, -1, self.parche * C)) + self.pe
        tok = (self.tok_modelo(modelo) + self.tok_edad(edad[:, None]))[:, None]
        h = self.enc(torch.cat([tok, z], 1))
        return self.cab(self.norm(h[:, 0]), modelo, edad)


def crear_modelo(arq, c_in, L, hidden=64, dropout=0.2):
    if arq == "Transformer":
        return TransformerNet(c_in, L, hidden, 4, 2, 4, dropout)
    if arq == "GRU":
        return GRUNet(c_in, hidden, 1, dropout)
    if arq == "TCN":
        return TCN(c_in, hidden // 2, L, 3, dropout)
    if arq == "CNN-LSTM":
        return CNNLSTM(c_in, hidden, dropout)
    raise ValueError(arq)


# perdidas

def focal_bin(logit, y, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * ce).mean()


def focal_multi(logits, y, gamma=2.0):
    logp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - logp.exp()) ** gamma * logp).mean()


def perdida(logit_b, logits_m, yb, ym, estrategia, pesos, lam):
    """Perdida combinada, binaria + lam * multiclase, segun la estrategia de desbalance."""
    if estrategia == "ponderada":
        lb = F.binary_cross_entropy_with_logits(logit_b, yb, pos_weight=pesos["pos"])
        lm = F.cross_entropy(logits_m, ym, weight=pesos["clases"])
    elif estrategia == "focal":
        lb, lm = focal_bin(logit_b, yb), focal_multi(logits_m, ym)
    else:  # submuestreo, el lote ya viene balanceado y se usa la perdida sin pesos
        lb = F.binary_cross_entropy_with_logits(logit_b, yb)
        lm = F.cross_entropy(logits_m, ym)
    return lb + lam * lm


# entrenamiento y evaluacion

@torch.no_grad()
def predecir(modelo, D, idx, L, bs=2048):
    modelo.eval()
    pb, pm = [], []
    for i in range(0, len(idx), bs):
        x, mo, ed, _, _ = lote(D, idx[i:i + bs], L)
        lb, lm = modelo(x, mo, ed)
        pb.append(torch.sigmoid(lb).cpu())
        pm.append(torch.softmax(lm, 1).cpu())
    return torch.cat(pb).numpy(), torch.cat(pm).numpy()


def entrenar(D, arq="GRU", L=72, estrategia="ponderada", hidden=64, dropout=0.2, lr=1e-3,
             weight_decay=1e-4, lam=0.5, epocas=10, muestras_epoca=40000, paciencia=3,
             bs=256, seed=42, verbose=True):
    """Entrena con early stopping sobre la PR-AUC de una submuestra temporal de validacion.

    La submuestra toma una de cada 4 horas de validacion, conserva la prevalencia real
    y hace que cada evaluacion sea rapida. Las metricas finales se calculan con toda la validacion.
    """
    fijar_semilla(seed)
    rng = np.random.default_rng(seed)
    c_in = D["X"].shape[2]
    modelo = crear_modelo(arq, c_in, L, hidden, dropout).to(DEVICE)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)

    itr = D["idx_train"]
    ytr = D["y_bin"][itr[:, 0], itr[:, 1]].numpy()
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    ymt = D["y_multi"][itr[:, 0], itr[:, 1]].numpy()
    frec = np.bincount(ymt, minlength=5) / len(ymt)
    pesos = {"pos": torch.tensor(len(neg) / len(pos), device=DEVICE),
             "clases": torch.tensor((1 / frec) / (1 / frec).mean(), dtype=torch.float32, device=DEVICE) ** 0.5}
    iv = D["idx_val"]
    iv_es = iv[iv[:, 1] % 4 == 0]
    yv_es = D["y_bin"][iv_es[:, 0], iv_es[:, 1]].numpy()

    hist, mejor, estado, sin_mejora = [], -1, None, 0
    for ep in range(epocas):
        t0 = time.time()
        if estrategia == "submuestreo":  # todas las ventanas pre-falla posibles y 4 sanas por cada una
            n_pos = min(len(pos), muestras_epoca // 5)
            sel = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, 4 * n_pos, replace=False)])
        else:  # muestra aleatoria con la prevalencia natural
            sel = rng.choice(len(itr), muestras_epoca, replace=False)
        rng.shuffle(sel)
        modelo.train()
        tot = 0.0
        for i in range(0, len(sel), bs):
            x, mo, ed, yb, ym = lote(D, itr[sel[i:i + bs]], L)
            lb, lm = modelo(x, mo, ed)
            loss = perdida(lb, lm, yb, ym, estrategia, pesos, lam)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
        pb, _ = predecir(modelo, D, iv_es, L)
        ap = average_precision_score(yv_es, pb)
        hist.append({"epoca": ep + 1, "perdida_train": tot / len(sel), "pr_auc_val": ap, "seg": time.time() - t0})
        if verbose:
            print(f"  época {ep + 1}  pérdida {tot / len(sel):.4f}  PR-AUC val {ap:.3f}  {time.time() - t0:.0f} s")
        if ap > mejor + 1e-4:
            mejor, sin_mejora = ap, 0
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(estado)
    return modelo, pd.DataFrame(hist)


def recall_a_precision(y, p, pmin=0.80):
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= pmin)[0]
    if len(ok) == 0:
        return 0.0, np.nan
    i = ok[np.argmax(rec[ok])]
    return rec[i], thr[i]


def umbral_alerta(y, p):
    """Umbral de alerta, el que maximiza el F1 de la tarea binaria en validacion."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
    return thr[np.argmax(f1)]


def tiempo_anticipacion(D, idx, p, fallas, umbral, ventana=72, sostenida=3):
    """Horas entre la primera alerta sostenida (p >= umbral durante 3 horas seguidas) y la falla.

    Para cada evento de falla del periodo se revisan las 72 horas previas. Devuelve la
    anticipacion de cada evento detectado y la fraccion de eventos detectados.
    """
    serie = np.full(D["y_bin"].shape, np.nan, dtype=np.float32)
    serie[idx[:, 0], idx[:, 1]] = p
    pos_hora = {h: i for i, h in enumerate(pd.to_datetime(D["horas"]))}
    pos_maq = {m: i for i, m in enumerate(D["maquinas"])}
    antic, total = [], 0
    for (m, f) in fallas[["machineID", "datetime"]].drop_duplicates().itertuples(index=False):
        if f not in pos_hora:
            continue
        i_f, im = pos_hora[f], pos_maq[m]
        if i_f == 0 or np.isnan(serie[im, i_f - 1]):
            continue  # solo fallas cuya ventana positiva cae en el periodo evaluado
        tramo = serie[im, max(0, i_f - ventana):i_f]
        total += 1
        alerta = np.nan_to_num(tramo, nan=0) >= umbral
        corrida = np.convolve(alerta.astype(int), np.ones(sostenida, int), "valid") == sostenida
        if corrida.any():
            antic.append(len(tramo) - np.argmax(corrida))
    return np.array(antic), (len(antic) / total if total else np.nan), total


def evaluar(D, idx, pb, pm, fallas, nombre=""):
    """Metricas completas en validacion para un modelo.

    La regla de decision usa las dos salidas, hay alerta si la probabilidad binaria supera el
    umbral de mejor F1 y el componente es el de mayor probabilidad en la salida multiclase.
    """
    yb = D["y_bin"][idx[:, 0], idx[:, 1]].numpy()
    ym = D["y_multi"][idx[:, 0], idx[:, 1]].numpy()
    r80, _ = recall_a_precision(yb, pb)
    u = umbral_alerta(yb, pb)
    antic, frac, _ = tiempo_anticipacion(D, idx, pb, fallas, u)
    pred_m = np.where(pb >= u, pm[:, 1:].argmax(1) + 1, 0)
    f1c = f1_score(ym, pred_m, labels=[1, 2, 3, 4], average=None, zero_division=0)
    res = {"modelo": nombre, "PR-AUC": average_precision_score(yb, pb), "ROC-AUC": roc_auc_score(yb, pb),
           "recall P≥0.80": r80, "F1 macro comp": f1c.mean(),
           "anticipación media (h)": antic.mean() if len(antic) else np.nan,
           "fallas detectadas %": 100 * frac, "Brier": brier_score_loss(yb, pb)}
    for c, v in zip(COMPONENTES, f1c):
        res[f"F1 {c}"] = v
    return res


# calibracion e interpretabilidad

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ajustar_temperatura(p, y):
    """Busca la temperatura T que minimiza la log-loss de sigmoid(logit(p) / T) en validacion."""
    z = torch.tensor(logit(p), dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def cierre():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / logT.exp(), yt)
        loss.backward()
        return loss
    opt.step(cierre)
    return float(logT.exp())


def aplicar_temperatura(p, T):
    return 1 / (1 + np.exp(-logit(p) / T))


def gradientes_integrados(modelo, D, idx, L, pasos=32, bs=64):
    """Integrated gradients del logit binario respecto a la ventana de entrada.

    La linea de referencia es una ventana de ceros, que equivale a la media de entrenamiento
    porque los canales estan estandarizados. Devuelve un arreglo (n, L, C).
    """
    modelo.eval()
    salida = []
    alfas = torch.linspace(0, 1, pasos + 1, device=DEVICE)[1:]
    for i in range(0, len(idx), bs):
        x, mo, ed, _, _ = lote(D, idx[i:i + bs], L)
        tot = torch.zeros_like(x)
        for a in alfas:
            xa = (a * x).detach().requires_grad_(True)
            with torch.backends.cudnn.flags(enabled=False):
                lb, _ = modelo(xa, mo, ed)
            g, = torch.autograd.grad(lb.sum(), xa)
            tot += g
        salida.append((x * tot / pasos).detach().cpu())
    return torch.cat(salida).numpy()


# politicas de mantenimiento

COSTOS = {"falla": 15000, "acierto": 3000, "falsa": 600}


def eventos_falla(D, idx, fallas):
    """Eventos de falla (maquina, hora) cuya ventana positiva cae dentro del periodo evaluado."""
    en_periodo = np.zeros(D["y_bin"].shape, bool)
    en_periodo[idx[:, 0], idx[:, 1]] = True
    pos_hora = {h: i for i, h in enumerate(pd.to_datetime(D["horas"]))}
    pos_maq = {m: i for i, m in enumerate(D["maquinas"])}
    ev = []
    for (m, f), g in fallas.groupby(["machineID", "datetime"]):
        if f in pos_hora and pos_hora[f] > 0 and en_periodo[pos_maq[m], pos_hora[f] - 1]:
            ev.append((pos_maq[m], pos_hora[f], tuple(sorted(g.failure))))
    return ev


def politica_modelo(D, idx, p, umbral, eventos, costos=COSTOS, sostenida=3, espera=24):
    """Simula la politica de alertas del modelo en un periodo.

    Se emite una alerta cuando la probabilidad supera el umbral 3 horas seguidas. Cada alerta
    dispara una intervencion. Si hay una falla de la maquina en las 24 h siguientes la intervencion
    es acertada y la falla se evita, si no es una falsa alarma. Despues de una alerta la maquina
    no puede volver a alertar durante 24 h. Las fallas sin intervencion previa son no anticipadas.
    """
    serie = np.full(D["y_bin"].shape, np.nan, dtype=np.float32)
    serie[idx[:, 0], idx[:, 1]] = p
    fallas_maq = {}
    for (im, i_f, comps) in eventos:
        fallas_maq.setdefault(im, []).append(i_f)
    aciertos, falsas, evitadas, antic, alertas = 0, 0, set(), [], []
    for im in range(serie.shape[0]):
        fila = serie[im]
        validas = np.where(~np.isnan(fila))[0]
        if len(validas) == 0:
            continue
        racha, libre = 0, -1
        fm = np.array(sorted(fallas_maq.get(im, [])))
        for t in validas:
            racha = racha + 1 if fila[t] >= umbral else 0
            if racha >= sostenida and t >= libre:
                libre = t + espera
                prox = fm[(fm > t) & (fm <= t + 24)] if len(fm) else []
                alertas.append((im, int(t), bool(len(prox))))
                if len(prox):
                    aciertos += 1
                    for f in prox:
                        evitadas.add((im, int(f)))
                    antic.append(prox[0] - t)
                else:
                    falsas += 1
    n = len(eventos)
    no_antic = sum(1 for (im, i_f, _) in eventos if (im, i_f) not in evitadas)
    costo = no_antic * costos["falla"] + aciertos * costos["acierto"] + falsas * costos["falsa"]
    return {"fallas": n, "evitadas": n - no_antic, "no anticipadas": no_antic, "aciertos": aciertos,
            "falsas alarmas": falsas, "costo": costo, "anticipación media (h)": np.mean(antic) if antic else np.nan,
            "alertas": alertas}


def politica_hasta_falla(eventos, costos=COSTOS):
    n = len(eventos)
    return {"fallas": n, "evitadas": 0, "no anticipadas": n, "aciertos": 0, "falsas alarmas": 0,
            "costo": n * costos["falla"], "anticipación media (h)": np.nan}


def politica_calendario(D, idx, man, eventos, dias, costos=COSTOS):
    """Reemplazo por calendario de cada componente cada `dias` dias desde su ultimo reemplazo.

    El ultimo reemplazo se toma de los registros de mantenimiento anteriores al periodo y despues
    se reinicia con cada reemplazo programado o correctivo. Un reemplazo programado evita una falla
    del mismo componente si esta ocurre en las 24 h siguientes (se cobra como intervencion
    acertada) y si no se cobra como intervencion sin falla.
    """
    horas = pd.to_datetime(D["horas"])
    t_ini, t_fin = idx[:, 1].min(), idx[:, 1].max()
    pos_maq = {m: i for i, m in enumerate(D["maquinas"])}
    paso = int(dias * 24)
    fallas_mc = {}
    for (im, i_f, comps) in eventos:
        for c in comps:
            fallas_mc.setdefault((im, c), []).append(i_f)
    aciertos, falsas, evitadas = 0, 0, set()
    man = man[man.datetime < horas[t_ini]]
    ult = man.groupby(["machineID", "comp"]).datetime.max()
    pos_hora = pd.Series(np.arange(len(horas)), index=horas)
    for m, im in pos_maq.items():
        for c in COMPONENTES:
            u = ult.get((m, c), horas[0])
            last = int(np.searchsorted(horas.values, np.datetime64(u)))
            fm = sorted(fallas_mc.get((im, c), []))
            t = last + paso
            while True:
                # una falla anterior al siguiente reemplazo programado reinicia el reloj
                previas = [f for f in fm if last < f <= t and (im, f, c) not in evitadas]
                if previas:
                    last = previas[0]
                    t = last + paso
                    continue
                if t > t_fin:
                    break
                if t >= t_ini:
                    prox = [f for f in fm if t < f <= t + 24]
                    if prox:
                        aciertos += 1
                        for f in prox:
                            evitadas.add((im, f, c))
                    else:
                        falsas += 1
                last = t
                t = last + paso
    n = len(eventos)
    no_antic = sum(1 for (im, i_f, comps) in eventos if not all((im, i_f, c) in evitadas for c in comps))
    costo = no_antic * costos["falla"] + aciertos * costos["acierto"] + falsas * costos["falsa"]
    return {"fallas": n, "evitadas": n - no_antic, "no anticipadas": no_antic, "aciertos": aciertos,
            "falsas alarmas": falsas, "costo": costo, "anticipación media (h)": np.nan}


def elegir_umbral(D, idx, p, eventos, costos=COSTOS, n=120):
    """Umbral que minimiza el costo esperado de la politica del modelo en el periodo dado."""
    cand = np.unique(np.quantile(p, np.linspace(0.90, 0.9995, n)))
    res = [(u, politica_modelo(D, idx, p, u, eventos, costos)["costo"]) for u in cand]
    return min(res, key=lambda r: r[1])[0], pd.DataFrame(res, columns=["umbral", "costo"])
