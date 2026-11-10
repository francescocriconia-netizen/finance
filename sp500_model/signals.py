"""Catalogo dei segnali: evidenza storica di ogni singola variabile, presa da sola.

Per i flag di calendario/evento a 1 giorno: rendimento medio dell'S&P 500 nei giorni
con l'evento rispetto agli altri giorni. Per le altre variabili: correlazione di rango
(IC) con il rendimento futuro. È un'analisi descrittiva sull'intero campione (non una
validazione fuori campione): con oltre cento segnali, alcuni risultano "significativi"
per puro caso, quindi conta soprattutto la coerenza tra il campione intero e il 2015+.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import group_of


def _one(x: pd.Series, y: pd.Series, h: int) -> dict:
    d = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    if len(d) < 250 or d["x"].nunique() < 2:
        return {}
    if h == 1 and set(d["x"].unique()) <= {0.0, 1.0}:
        on, off = d.y[d.x > 0], d.y[d.x == 0]
        if len(on) < 20:
            return {}
        diff = on.mean() - off.mean()
        t = diff / np.sqrt(on.var() / len(on) + off.var() / len(off))
        return {"tipo": "evento", "n": int(len(on)), "effetto": float(diff), "t": float(t),
                "media_con_evento": float(on.mean()), "media_altri_giorni": float(off.mean())}
    ic = spearmanr(d.x, d.y)[0]
    n_eff = len(d) / h
    return {"tipo": "ic", "n": int(len(d)), "effetto": float(ic), "t": float(ic * np.sqrt(max(n_eff - 2, 1)) / np.sqrt(1 - ic ** 2))}


def catalogue(X: dict, Y: pd.DataFrame, horizons) -> pd.DataFrame:
    rows = []
    for h in horizons:
        x, y = X[h], Y[h]
        recent = x.index.year >= 2015
        for c in x.columns:
            full = _one(x[c], y, h)
            if not full:
                continue
            rec = _one(x[c][recent], y[recent], h)
            rows.append({"orizzonte": h, "variabile": c, "gruppo": group_of(c), **full,
                         "effetto_2015": rec.get("effetto", np.nan), "t_2015": rec.get("t", np.nan),
                         "n_2015": rec.get("n", 0)})
    out = pd.DataFrame(rows)
    out["coerente"] = np.sign(out["effetto"]) == np.sign(out["effetto_2015"])
    return out.sort_values(["orizzonte", "t"], key=lambda s: s.abs() if s.name == "t" else s, ascending=[True, False])
