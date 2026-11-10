"""Addestramento walk-forward, metriche fuori campione, ablation per gruppi e previsione finale.

Schema di validazione: per ogni anno di test Y il modello è addestrato solo su dati
precedenti a Y (con un "embargo" di h giorni per non far toccare le etichette di
train con il periodo di test). Gli iperparametri sono fissati a priori e mai
ottimizzati sul periodo di test.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from .features import GROUPS, group_of

warnings.filterwarnings("ignore", category=UserWarning)

LGB_PARAMS = dict(n_estimators=200, learning_rate=0.02, num_leaves=4, max_depth=2, subsample=0.7, subsample_freq=1,
                  colsample_bytree=0.3, reg_lambda=20.0, verbose=-1, n_jobs=4)
ALPHAS = np.logspace(1, 5, 9)
TEST_START = 2005


def lgb_params(h: int) -> dict:
    """Alberi piccolissimi; ogni foglia deve contenere almeno ~40 finestre indipendenti di h giorni."""
    return dict(LGB_PARAMS, min_child_samples=max(200, 40 * h))


class Ridge:
    """Ridge su variabili standardizzate (mediana per i mancanti, valori estremi troncati a ±5 sigma)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray, h: int):
        self.cols = list(X.columns)
        self.med = X.median()
        Z = X.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1.0)
        Z = ((Z - self.mu) / self.sd).clip(-5, 5).fillna(0.0)
        self.m = RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5, gap=h)).fit(Z.values, y)
        return self

    def _z(self, X):
        return ((X[self.cols].fillna(self.med) - self.mu) / self.sd).clip(-5, 5).fillna(0.0)

    def predict(self, X):
        return self.m.predict(self._z(X).values)

    def contributions(self, X):
        return self._z(X) * self.m.coef_


class Blend:
    """Media tra gradient boosting (LightGBM, più semi) e ridge; opzionalmente probabilità di rialzo e quantili."""

    def __init__(self, h: int, seeds=(0, 1, 2), extras: bool = True):
        self.h, self.seeds, self.extras = h, seeds, extras

    def fit(self, X: pd.DataFrame, y: pd.Series):
        lo, hi = y.quantile([0.01, 0.99])
        yw = y.clip(lo, hi).values                      # target troncato all'1%: meno peso ai giorni di crollo
        self.mean_ = float(y.mean())
        P = lgb_params(self.h)
        self.ridge = Ridge().fit(X, yw, self.h)
        self.gbm = [lgb.LGBMRegressor(**P, random_state=s).fit(X, yw) for s in self.seeds]
        if self.extras:
            self.clf = lgb.LGBMClassifier(**P, random_state=0).fit(X, (y.values > 0).astype(int))
            self.q = {a: lgb.LGBMRegressor(**P, objective="quantile", alpha=a, random_state=0).fit(X, y.values)
                      for a in (0.1, 0.9)}
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        out["pred_gbm"] = np.mean([m.predict(X) for m in self.gbm], axis=0)
        out["pred_ridge"] = self.ridge.predict(X)
        out["pred"] = 0.5 * (out["pred_gbm"] + out["pred_ridge"])
        out["bench"] = self.mean_
        if self.extras:
            out["prob_up"] = self.clf.predict_proba(X)[:, 1]
            out["q10"], out["q90"] = self.q[0.1].predict(X), self.q[0.9].predict(X)
        return out

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contributo di ogni variabile alla previsione (TreeSHAP per il boosting, coef*z per la ridge)."""
        g = np.mean([m.predict(X, pred_contrib=True)[:, :-1] for m in self.gbm], axis=0)
        r = self.ridge.contributions(X).values
        return pd.DataFrame(0.5 * (g + r), index=X.index, columns=X.columns)

    def importance(self) -> pd.Series:
        gain = np.mean([m.booster_.feature_importance("gain") for m in self.gbm], axis=0)
        return pd.Series(gain / gain.sum(), index=self.ridge.cols)


def walk_forward(X: pd.DataFrame, y: pd.Series, h: int, cols=None, seeds=(0, 1, 2), extras=True,
                 test_start: int = TEST_START) -> pd.DataFrame:
    """Previsioni fuori campione anno per anno (finestra di train espandibile)."""
    Xc = X if cols is None else X[cols]
    idx = X.index
    out = []
    for year in range(test_start, idx.year.max() + 1):
        test = idx.year == year
        first = np.argmax(test)
        train = np.zeros(len(idx), dtype=bool)
        train[: max(first - h, 0)] = True                  # embargo di h giorni prima del test
        train &= y.notna().values
        model = Blend(h, seeds, extras).fit(Xc[train], y[train])
        p = model.predict(Xc[test])
        p["y"] = y[test]
        out.append(p)
    return pd.concat(out)


def evaluate(oos: pd.DataFrame, h: int) -> dict:
    """Metriche fuori campione (solo dove il rendimento realizzato è noto)."""
    d = oos.dropna(subset=["y"])
    n_eff = len(d) / h                                     # osservazioni indipendenti (finestre sovrapposte)
    ic = spearmanr(d["pred"], d["y"])[0]
    res = {
        "n_giorni": int(len(d)), "dal": str(d.index.min().date()), "al": str(d.index.max().date()),
        "r2_oos": float(1 - ((d.y - d.pred) ** 2).sum() / ((d.y - d.bench) ** 2).sum()),
        "ic": float(ic), "ic_tstat": float(ic * np.sqrt(max(n_eff - 2, 1)) / np.sqrt(1 - ic ** 2)),
        "hit_rate": float((np.sign(d.pred) == np.sign(d.y)).mean()), "quota_rialzi": float((d.y > 0).mean()),
        "rend_medio_se_pred_pos": float(d.y[d.pred > 0].mean()), "rend_medio_se_pred_neg": float(d.y[d.pred <= 0].mean()),
        "quota_pred_neg": float((d.pred <= 0).mean()),
    }
    q = pd.qcut(d["pred"].rank(method="first"), 5, labels=False)
    res["rend_medio_per_quintile"] = [float(v) for v in d.groupby(q)["y"].mean()]
    if "prob_up" in d:
        hi, lo = d.prob_up >= d.prob_up.quantile(0.8), d.prob_up <= d.prob_up.quantile(0.2)
        res["rialzi_se_prob_alta"], res["rialzi_se_prob_bassa"] = float((d.y[hi] > 0).mean()), float((d.y[lo] > 0).mean())
        res["copertura_q10_q90"] = float(((d.y >= d.q10) & (d.y <= d.q90)).mean())
    return res


def backtest(oos: pd.DataFrame, px: pd.Series, cost_bps: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Strategia semplice: investito nell'S&P il giorno dopo se la previsione è > 0, altrimenti liquidità (rendimento 0)."""
    r = px.pct_change().reindex(oos.index)
    pos = (oos["pred"] > 0).astype(float).shift(1)        # segnale alla chiusura di t, posizione su t+1
    df = pd.DataFrame({"mercato": r, "pos": pos}).dropna()
    df["strategia"] = df["pos"] * df["mercato"] - df["pos"].diff().abs().fillna(0) * cost_bps / 1e4

    def stats(x):
        eq = (1 + x).cumprod()
        yrs = len(x) / 252
        return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(x.std() * np.sqrt(252)),
                "sharpe": float(x.mean() / x.std() * np.sqrt(252)), "max_drawdown": float((eq / eq.cummax() - 1).min())}

    return df, {"strategia": stats(df["strategia"]), "buy_and_hold": stats(df["mercato"]),
                "tempo_investito": float(df["pos"].mean()), "n_cambi": int(df["pos"].diff().abs().sum())}


PERIODS = {"sviluppo": (2005, 2014), "verifica": (2015, 2100)}


def by_period(oos: pd.DataFrame, h: int) -> dict:
    """Metriche sull'intero fuori campione e sui due sotto-periodi (sviluppo 2005-14, verifica 2015+)."""
    out = {"totale": evaluate(oos, h)}
    for name, (a, b) in PERIODS.items():
        out[name] = evaluate(oos[(oos.index.year >= a) & (oos.index.year <= b)], h)
    return out


def ablation(X: pd.DataFrame, y: pd.Series, h: int) -> pd.DataFrame:
    """Quanto vale ogni gruppo: modello con il solo gruppo e modello senza quel gruppo (IC e R² fuori campione)."""
    rows = []
    for g in GROUPS:
        own = [c for c in X.columns if group_of(c) == g]
        rest = [c for c in X.columns if group_of(c) != g]
        for label, cols in (("solo", own), ("senza", rest)):
            m = by_period(walk_forward(X, y, h, cols=cols, seeds=(0,), extras=False), h)
            row = {"orizzonte": h, "gruppo": g, "variante": label, "n_variabili": len(cols)}
            for per, mm in m.items():
                row[f"ic_{per}"], row[f"t_{per}"], row[f"r2_{per}"] = mm["ic"], mm["ic_tstat"], mm["r2_oos"]
            rows.append(row)
    return pd.DataFrame(rows)


def select_groups(abl: pd.DataFrame, min_t: float = 1.0) -> list[str]:
    """Regola fissata a priori: tengo i gruppi che, da soli, hanno IC positivo (t >= 1) nel periodo di sviluppo.

    Il periodo di verifica (2015+) non entra nella scelta: resta un banco di prova pulito.
    Se nessun gruppo supera la soglia la lista è vuota: per quell'orizzonte il modello non ha
    dimostrato abilità e la previsione puntuale torna alla media storica.
    """
    solo = abl[abl["variante"] == "solo"]
    return solo[(solo["ic_sviluppo"] > 0) & (solo["t_sviluppo"] >= min_t)]["gruppo"].tolist()


def final_forecast(X: pd.DataFrame, y: pd.Series, h: int, asof: pd.Timestamp):
    """Modello finale su tutto lo storico etichettato e previsione alla data `asof`."""
    lab = y.notna()
    model = Blend(h).fit(X[lab], y[lab])
    row = X.loc[[asof]]
    pred = model.predict(row).iloc[0]
    contrib = model.contributions(row).iloc[0].sort_values()
    return model, pred, contrib
