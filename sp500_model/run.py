"""Esegue tutta la pipeline: dati -> variabili -> validazione walk-forward -> previsione.

    python -m sp500_model.run                      # usa i dati salvati in data/tidy
    python -m sp500_model.run --refresh            # riscarica le fonti
    python -m sp500_model.run --last-price 7549.94 --last-price-date 2026-11-09

I risultati finiscono in sp500_model/output/.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import model as M
from . import sources as S
from .signals import catalogue

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "output"
HORIZONS = (1, 5, 21)
LABEL = {1: "1 giorno", 5: "1 settimana", 21: "1 mese"}


def _round(o, nd=6):
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), nd)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="riscarica tutte le fonti")
    ap.add_argument("--today", default=None, help="data di riferimento (default: oggi)")
    ap.add_argument("--last-price", type=float, default=None, help="ultima chiusura nota dell'S&P 500 (ancora per i livelli di prezzo)")
    ap.add_argument("--last-price-date", default=None, help="data dell'ultima chiusura nota")
    ap.add_argument("--skip-ablation", action="store_true", help="riusa output/ablation.csv se presente")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    print("1) dati")
    data = S.load_all(refresh=a.refresh)
    X, Y, px, asof = F.build_dataset(data, HORIZONS, today=a.today)
    anchor_px = a.last_price or float(px.iloc[-1])
    anchor_dt = pd.Timestamp(a.last_price_date) if a.last_price_date else px.index[-1]
    print(f"   {X[1].shape[1]} variabili, {len(X[1])} giorni; previsione alla chiusura del {asof.date()}; "
          f"ancora di prezzo {anchor_px:,.2f} ({anchor_dt.date()})")

    # freschezza dei dati alla data della previsione
    fresh = pd.DataFrame({"gruppo": [F.group_of(c) for c in X[1].columns],
                          "ultimo_dato": X[1].apply(lambda c: c.last_valid_index()),
                          "disponibile_oggi": X[1].loc[asof].notna()})
    fresh.to_csv(OUT / "freshness.csv", index_label="variabile")

    catalogue(X, Y, HORIZONS).to_csv(OUT / "signals.csv", index=False, float_format="%.6g")

    metrics, forecast, oos_all, eq_all, imp_all = {}, {}, [], [], []
    abl_path = OUT / "ablation.csv"
    abl_prev = pd.read_csv(abl_path) if (a.skip_ablation and abl_path.exists()) else None
    abl_all = []

    for h in HORIZONS:
        Xh, yh = X[h], Y[h]
        print(f"2) orizzonte {LABEL[h]}: validazione walk-forward ({time.time() - t0:.0f}s)")
        oos_full = M.walk_forward(Xh, yh, h)
        if abl_prev is not None:
            abl = abl_prev[abl_prev["orizzonte"] == h]
        else:
            print("   contributo dei gruppi (ablation)")
            abl = M.ablation(Xh, yh, h)
        abl_all.append(abl)
        groups = M.select_groups(abl)
        skill = len(groups) > 0
        cols = [c for c in Xh.columns if F.group_of(c) in groups] if skill else list(Xh.columns)
        print(f"   gruppi selezionati sul periodo di sviluppo: {groups or 'nessuno (previsione = media storica)'}")
        pure = [c for c in Xh.columns if F.group_of(c) in F.NO_MARKET_GROUPS]
        variants = {"tutte_le_variabili": oos_full, "senza_prezzi_di_mercato": M.walk_forward(Xh, yh, h, cols=pure, extras=False)}
        if skill:
            variants["selezionato"] = M.walk_forward(Xh, yh, h, cols=cols) if len(groups) < len(F.GROUPS) else oos_full

        metrics[h] = {"orizzonte": LABEL[h], "gruppi_selezionati": groups, "n_variabili_selezionate": len(cols) if skill else 0}
        for name, oos in variants.items():
            bt_df, bt = M.backtest(oos, px)
            bt_ver = M.backtest(oos[oos.index.year >= M.PERIODS["verifica"][0]], px)[1]
            metrics[h][name] = {**M.by_period(oos, h), "backtest": bt, "backtest_verifica": bt_ver}
            eq_all.append(pd.DataFrame({"orizzonte": h, "modello": name, "mercato": (1 + bt_df["mercato"]).cumprod(),
                                        "strategia": (1 + bt_df["strategia"]).cumprod(), "pos": bt_df["pos"]}))
            o = oos.copy()
            o["orizzonte"], o["modello"] = h, name
            oos_all.append(o)

        print("   modello finale e previsione")
        model, pred, contrib = M.final_forecast(Xh[cols], yh, h, asof)
        base_up = float((yh.dropna() > 0).mean())
        if not skill:  # nessuna abilità dimostrata: niente inclinazione, restano solo media storica e ampiezza dell'intervallo
            pred = pred.copy()
            pred["pred"], pred["prob_up"] = pred["bench"], base_up
            contrib = contrib * 0.0
        by_group = contrib.groupby(contrib.index.map(F.group_of)).sum()
        exp_ret = float(pred["pred"])
        forecast[h] = {
            "orizzonte": LABEL[h], "giorni_di_borsa": h, "abilita_dimostrata": skill, "gruppi_usati": groups,
            "rendimento_atteso": exp_ret, "rendimento_medio_storico": float(pred["bench"]),
            "prob_rialzo": float(pred["prob_up"]), "quota_storica_rialzi": base_up, "q10": float(pred["q10"]), "q90": float(pred["q90"]),
            "prezzo_atteso": anchor_px * float(np.exp(exp_ret)),
            "prezzo_q10": anchor_px * float(np.exp(pred["q10"])), "prezzo_q90": anchor_px * float(np.exp(pred["q90"])),
            "contributo_gruppi": {g: float(v) for g, v in by_group.sort_values().items()} if skill else {},
            "spinte_al_rialzo": {k: float(v) for k, v in contrib.tail(6)[::-1].items() if v > 0},
            "spinte_al_ribasso": {k: float(v) for k, v in contrib.head(6).items() if v < 0},
        }
        imp = model.importance().rename("importanza").to_frame()
        imp["orizzonte"], imp["gruppo"] = h, imp.index.map(F.group_of)
        imp_all.append(imp)

    pd.concat(abl_all).to_csv(abl_path, index=False, float_format="%.5f")
    pd.concat(oos_all).to_csv(OUT / "oos_predictions.csv.gz", index_label="date", float_format="%.6f")
    pd.concat(eq_all).to_csv(OUT / "equity.csv.gz", index_label="date", float_format="%.5f")
    pd.concat(imp_all).to_csv(OUT / "importance.csv", index_label="variabile", float_format="%.5f")
    meta = {"generato_il": str(pd.Timestamp.today().date()), "previsione_alla_chiusura_del": str(asof.date()),
            "ultimo_prezzo": anchor_px, "data_ultimo_prezzo": str(anchor_dt.date()),
            "storico_prezzi_fino_al": str(px.index[-1].date()), "n_variabili": int(X[1].shape[1]),
            "train_dal": str(X[1].index[0].date()), "test_dal": M.TEST_START,
            "variabili_non_aggiornate": sorted(fresh.index[~fresh["disponibile_oggi"]].tolist())}
    (OUT / "metrics.json").write_text(json.dumps(_round({"meta": meta, "orizzonti": metrics}), indent=1, ensure_ascii=False))
    (OUT / "forecast.json").write_text(json.dumps(_round({"meta": meta, "previsioni": forecast}), indent=1, ensure_ascii=False))

    print(f"\nPrevisione S&P 500 dalla chiusura del {asof.date()} (ultimo prezzo {anchor_px:,.2f}):")
    for h in HORIZONS:
        f = forecast[h]
        print(f"  {f['orizzonte']:12s} atteso {f['rendimento_atteso']:+.2%} -> {f['prezzo_atteso']:,.0f} "
              f"(80%: {f['prezzo_q10']:,.0f} - {f['prezzo_q90']:,.0f}), prob. rialzo {f['prob_rialzo']:.0%}")
    print(f"fatto in {time.time() - t0:.0f}s; risultati in {OUT}")


if __name__ == "__main__":
    main()
