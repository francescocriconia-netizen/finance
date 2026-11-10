"""Fonti dati del modello S&P 500.

Ogni funzione `load_*` scarica la fonte grezza (cache in data/raw, non versionata),
la riduce a una tabella "tidy" indicizzata per data e la salva in data/tidy (versionata),
così la pipeline è riproducibile anche offline. `refresh=True` forza un nuovo download.

Nessuna fonte contiene lo storico dei prezzi dell'S&P 500 come *input*: il prezzo
(`load_spx`) è usato solo come variabile da prevedere.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
TIDY = ROOT / "data" / "tidy"
GH = "https://raw.githubusercontent.com/"
S3_STILES = "https://s3.us-west-1.amazonaws.com/stilesdata.com/trump-truth-social-archive/"
UA = {"User-Agent": "sp500-model/1.0 (research script)"}

# Mirror pubblici di serie FRED (id serie -> csv). A = storico completo, B = più aggiornato.
FRED_A = GH + "nickchen494949/macro-observer/HEAD/csv/fred/{id}.csv"
FRED_B = GH + "cool1990/invest-dashboard_v1/HEAD/data/series/{id}.csv"
MACRO_R = GH + "riverfjs/notification/HEAD/data/macro/{name}.csv"


# --------------------------------------------------------------------------- utilità
def fetch(url: str, name: str, refresh: bool = False, tries: int = 4) -> Path:
    """Scarica `url` in data/raw/`name` (se assente o refresh) e restituisce il percorso."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return path
    err = None
    for k in range(tries):
        try:
            with requests.get(url, headers=UA, timeout=180, stream=True) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(path)
            return path
        except Exception as e:  # rete instabile: riprova con attesa crescente
            err = e
            time.sleep(2 ** (k + 1))
    raise RuntimeError(f"download fallito: {url}: {err}")


def tidy(name: str):
    """Decorator: salva/riusa il risultato in data/tidy/<name>.csv."""

    def deco(fn):
        def wrapper(refresh: bool = False):
            TIDY.mkdir(parents=True, exist_ok=True)
            path = TIDY / f"{name}.csv"
            if path.exists() and not refresh:
                return pd.read_csv(path, index_col=0, parse_dates=True)
            df = fn(refresh)
            if isinstance(df, pd.Series):
                df = df.to_frame(df.name or name)
            df.index.name = "date"
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(path, float_format="%.6g")
            return df

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return deco


def _read_two_col(path: Path) -> pd.Series:
    """CSV (data, valore) con intestazioni qualsiasi -> Series numerica."""
    df = pd.read_csv(path, comment="#")
    df = df.iloc[:, :2]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    return df.set_index("date")["value"].sort_index()


def fred(series_id: str, refresh: bool = False, use_b: bool = True) -> pd.Series:
    """Serie FRED dai mirror pubblici: storico da A, coda più recente da B se presente."""
    parts = []
    try:
        parts.append(_read_two_col(fetch(FRED_A.format(id=series_id), f"fredA_{series_id}.csv", refresh)))
    except Exception:
        pass
    if use_b:
        try:
            parts.append(_read_two_col(fetch(FRED_B.format(id=series_id), f"fredB_{series_id}.csv", refresh, tries=1)))
        except Exception:
            pass
    if not parts:
        raise RuntimeError(f"serie FRED non disponibile: {series_id}")
    s = parts[0]
    for extra in parts[1:]:
        s = pd.concat([s, extra[extra.index > s.index.max()]])
    s.name = series_id
    return s


# --------------------------------------------------------------------------- target
@tidy("spx")
def load_spx(refresh=False):
    """S&P 500 giornaliero (solo come target). Fino al 2016 chiusure Yahoo ^GSPC, poi serie FRED SP500.

    Le due fonti coincidono sul periodo comune 2016-2018 (scarto massimo 0,0015%).
    """
    old = pd.read_csv(fetch(GH + "vijinho/sp500/master/csv/sp500.csv", "gspc_1950_2018.csv", refresh),
                      usecols=["Date", "Close"], parse_dates=["Date"]).set_index("Date")["Close"]
    new = fred("SP500", refresh, use_b=False)
    spx = pd.concat([old[old.index < new.index.min()], new])
    return spx.loc["1985":].rename("close")


@tidy("spx_monthly")
def load_spx_monthly(refresh=False):
    """S&P 500 mensile (media del mese, dataset Shiller): serve solo come controllo incrociato."""
    df = pd.read_csv(fetch(GH + "datasets/s-and-p-500/main/data/data.csv", "shiller.csv", refresh), parse_dates=["Date"])
    return df.set_index("Date")["SP500"].loc["1985":].rename("sp500_avg")


# --------------------------------------------------------------------------- mercati diversi dall'azionario
FRED_SERIES = ["WALCL", "WTREGEN", "RRPONTSYD", "WRESBAL", "ICSA", "DGS10", "DGS2", "DGS3MO", "T10YIE", "DFII10",
               "DBAA", "DAAA", "UMCSENT", "SAHMREALTIME", "WM2NS", "MMMFFAQ027S", "DRTSCILM"]


@tidy("fred")
def load_fred(refresh=False):
    """Serie FRED (liquidità Fed/Tesoro, tassi, credito, macro) in formato largo; NaN dove non osservate."""
    cols = {sid: fred(sid, refresh) for sid in FRED_SERIES}
    return pd.DataFrame(cols).loc["1985":]


@tidy("hy_oas")
def load_hy_oas(refresh=False):
    """Spread high yield (ICE BofA OAS): archivio storico + code recenti dai mirror FRED."""
    s = _read_two_col(fetch(MACRO_R.format(name="hy_oas_mirror"), "hy_oas_mirror.csv", refresh))
    try:
        tail = fred("BAMLH0A0HYM2", refresh)
        s = pd.concat([s, tail[tail.index > s.index.max()]])
    except Exception:
        pass
    return s.rename("hy_oas")


@tidy("vix")
def load_vix(refresh=False):
    df = pd.read_csv(fetch(GH + "datasets/finance-vix/main/data/vix-daily.csv", "vix-daily.csv", refresh), parse_dates=["DATE"])
    s = df.set_index("DATE")["CLOSE"]
    try:
        tail = _read_two_col(fetch(FRED_B.format(id="VIXCLS"), "fredB_VIXCLS.csv", refresh, tries=1))
        s = pd.concat([s, tail[tail.index > s.index.max()]])
    except Exception:
        pass
    return s.rename("vix")


@tidy("oil")
def load_oil(refresh=False):
    s = _read_two_col(fetch(GH + "datasets/oil-prices/main/data/wti-daily.csv", "wti-daily.csv", refresh))
    return s.rename("wti")


@tidy("gold")
def load_gold(refresh=False):
    s = _read_two_col(fetch(GH + "datasets/gold-prices/main/data/monthly.csv", "gold-monthly.csv", refresh))
    return s.loc["1984":].rename("gold")


@tidy("usd")
def load_usd(refresh=False):
    """Indice del dollaro 'fatto in casa': media dei log-cambi contro JPY, GBP, CAD, CHF, AUD (su = dollaro forte)."""
    df = pd.read_csv(fetch(GH + "datasets/exchange-rates/main/data/daily.csv", "fx-daily.csv", refresh), parse_dates=["Date"])
    df["Exchange rate"] = pd.to_numeric(df["Exchange rate"], errors="coerce")
    w = df.pivot_table(index="Date", columns="Country", values="Exchange rate")
    per_usd = ["Japan", "Canada", "Switzerland"]          # unità di valuta estera per 1 USD
    usd_per = ["United Kingdom", "Australia"]              # USD per 1 unità di valuta estera
    logs = pd.concat([np.log(w[c]) for c in per_usd] + [-np.log(w[c]) for c in usd_per], axis=1)
    idx = logs.diff().mean(axis=1, skipna=True).cumsum()
    return idx.loc["1984":].rename("usd_log")


# --------------------------------------------------------------------------- notizie
@tidy("epu")
def load_epu(refresh=False):
    """Economic Policy Uncertainty giornaliero (Baker-Bloom-Davis): quota di articoli su incertezza di politica economica."""
    df = pd.read_csv(fetch(GH + "androidilicious/finorchestra/HEAD/data/raw/epu/All_Daily_Policy_Data.csv", "epu_daily.csv", refresh))
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day), errors="coerce")
    return df.dropna(subset=["date"]).set_index("date")["daily_policy_index"].rename("epu")


@tidy("news_sentiment")
def load_news_sentiment(refresh=False):
    """Daily News Sentiment Index (Fed di San Francisco): tono degli articoli economici dei quotidiani USA."""
    s = _read_two_col(fetch(GH + "ramiuness/predict-brokerage-account-openings/HEAD/data/news_sentiment_data.csv", "news_sentiment.csv", refresh))
    return s.rename("news_sentiment")


@tidy("gpr")
def load_gpr(refresh=False):
    """Geopolitical Risk Index giornaliero (Caldara-Iacoviello), conteggio di articoli su tensioni geopolitiche."""
    s = _read_two_col(fetch(GH + "johnny-xiao-li/dual-narrative-10k/HEAD/dna/Data/gpr_daily.csv", "gpr_daily_a.csv", refresh))
    try:  # estensione più recente (stessa fonte, mirror diverso): usata solo se coerente sul periodo comune
        b = _read_two_col(fetch(GH + "BbangYi/geopolitical-risk-sector-dashboard/HEAD/data/processed/gpr_daily.csv", "gpr_daily_b.csv", refresh, tries=1))
        j = pd.concat([s, b], axis=1, keys=["a", "b"]).dropna()
        if len(j) > 500 and j.tail(750).corr().iloc[0, 1] > 0.98:
            k = (j.a.tail(750) / j.b.tail(750)).median()
            s = pd.concat([s, b[b.index > s.index.max()] * k])
    except Exception:
        pass
    return s.rename("gpr")


_TW = re.compile(r"^@realDonaldTrump, (\d{4}-\d{2}-\d{2}) \d{2}:\d{2}, \S+, (.*)$")
KW = {
    "mkt": re.compile(r"stock market|\bdow\b|nasdaq|s&p|wall street|401\(?k|record high|\bstocks\b", re.I),
    "trade": re.compile(r"tariff|trade deal|trade war|\bchina\b", re.I),
    "fed": re.compile(r"\bfed\b|federal reserve|powell|interest rate", re.I),
}


@tidy("trump_posts")
def load_trump(refresh=False):
    """Post giornalieri di Donald Trump (Twitter 2009-2021, Truth Social 2022-2025): volume e temi di mercato."""
    rows = []
    for nm in ("bf_office", "in_office"):
        p = fetch(GH + f"MarkHershey/CompleteTrumpTweetsArchive/HEAD/data/realDonaldTrump_{nm}.csv", f"trump_{nm}.csv", refresh)
        cur = None
        for line in open(p, encoding="utf-8", errors="ignore"):
            m = _TW.match(line.rstrip("\n"))
            if m:
                if cur:
                    rows.append(cur)
                cur = [m.group(1), m.group(2)]
            elif cur:
                cur[1] += " " + line.strip()
        if cur:
            rows.append(cur)
    tw = pd.DataFrame(rows, columns=["date", "text"])
    ts = pd.read_csv(fetch(S3_STILES + "truth_archive.csv", "truth_archive.csv", refresh), usecols=["created_at", "content"])
    ts = pd.DataFrame({"date": ts["created_at"].str[:10], "text": ts["content"].fillna("")})
    allp = pd.concat([tw, ts], ignore_index=True)
    allp["date"] = pd.to_datetime(allp["date"], errors="coerce")
    allp = allp.dropna(subset=["date"])
    out = pd.DataFrame({"posts": allp.groupby("date").size()})
    for k, rx in KW.items():
        out[k] = allp[allp["text"].str.contains(rx)].groupby("date").size()
    days = pd.date_range(out.index.min(), out.index.max(), freq="D")
    return out.reindex(days).fillna(0).astype(int)


# --------------------------------------------------------------------------- soggetti specifici / posizionamento
@tidy("cot")
def load_cot(refresh=False):
    """CFTC Traders in Financial Futures sull'E-mini S&P 500: netto di asset manager, fondi a leva e dealer."""
    df = pd.read_csv(fetch(MACRO_R.format(name="cot_sp500"), "cot_sp500.csv", refresh), parse_dates=["date"]).set_index("date")
    return df[["open_interest", "lev_money_net", "asset_mgr_net", "dealer_net"]]


@tidy("aaii")
def load_aaii(refresh=False):
    """Sondaggio AAII tra investitori individuali (quota rialzisti / ribassisti, in %)."""
    df = pd.read_csv(fetch(MACRO_R.format(name="aaii"), "aaii.csv", refresh), parse_dates=["date"]).set_index("date")
    return df[["bullish", "bearish"]]


@tidy("naaim")
def load_naaim(refresh=False):
    """NAAIM Exposure Index: esposizione azionaria media dei gestori attivi USA."""
    df = pd.read_csv(fetch(MACRO_R.format(name="naaim"), "naaim.csv", refresh), parse_dates=["Date"]).set_index("Date")
    return df["naaim_mean"].rename("naaim")


@tidy("putcall")
def load_putcall(refresh=False):
    """Rapporto put/call sulle azioni (CBOE): archivio 2006-2019 + serie corrente."""
    a = pd.read_csv(fetch(MACRO_R.format(name="putcall"), "putcall.csv", refresh), comment="#", parse_dates=["date"]).set_index("date")["equity_pc"]
    b = pd.read_csv(fetch(MACRO_R.format(name="putcall_cboe"), "putcall_cboe.csv", refresh), comment="#", parse_dates=["date"]).set_index("date")["equity_pc"]
    s = pd.concat([a.dropna(), b.dropna()])
    return s[~s.index.duplicated(keep="last")].rename("equity_pc")


_AMT = re.compile(r"\$([\d,]+)\s*-\s*\$([\d,]+)")


@tidy("senate")
def load_senate(refresh=False):
    """Compravendite di azioni dichiarate dai senatori USA (STOCK Act), per data dell'operazione."""
    d = json.load(open(fetch(GH + "timothycarambat/senate-stock-watcher-data/master/aggregate/all_transactions.json", "senate.json", refresh)))
    df = pd.DataFrame(d)
    df = df[df["asset_type"] == "Stock"].copy()
    df["date"] = pd.to_datetime(df["transaction_date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date"])

    def mid(a):
        m = _AMT.search(a or "")
        return (int(m.group(1).replace(",", "")) + int(m.group(2).replace(",", ""))) / 2 if m else np.nan

    df["usd"] = df["amount"].map(mid)
    buy = df[df["type"] == "Purchase"].groupby("date").agg(n_buy=("usd", "size"), usd_buy=("usd", "sum"))
    sell = df[df["type"].str.startswith("Sale")].groupby("date").agg(n_sell=("usd", "size"), usd_sell=("usd", "sum"))
    out = buy.join(sell, how="outer")
    days = pd.date_range(out.index.min(), out.index.max(), freq="D")
    return out.reindex(days).fillna(0)


# --------------------------------------------------------------------------- natura
NYC_STATION = "USW00094728"  # Central Park


@tidy("weather_nyc")
def load_weather(refresh=False):
    """Meteo giornaliero di New York Central Park (NOAA GHCN-Daily): pioggia, neve, temperature."""
    keep = {"PRCP", "SNOW", "TMAX", "TMIN"}
    rows = []
    p = fetch(f"https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{NYC_STATION}.csv", "ghcn_nyc.csv", refresh)
    with open(p) as f:
        for r in csv.reader(f):
            if r[2] in keep and r[1] >= "19800101":
                rows.append((r[1], r[2], r[3]))
    last = max(r[0] for r in rows)
    for year in range(int(last[:4]), pd.Timestamp.today().year + 1):  # anni recenti: file annuali (grandi) filtrati al volo
        url = f"https://noaa-ghcn-pds.s3.amazonaws.com/csv.gz/by_year/{year}.csv.gz"
        cache = RAW / f"ghcn_nyc_{year}.csv"
        if not cache.exists() or refresh or year >= pd.Timestamp.today().year:
            with requests.get(url, headers=UA, timeout=600, stream=True) as resp:
                resp.raise_for_status()
                gz = gzip.GzipFile(fileobj=resp.raw)
                with open(cache, "w") as out:
                    for raw in io.TextIOWrapper(gz, encoding="utf-8"):
                        if raw.startswith(NYC_STATION):
                            out.write(raw)
        with open(cache) as f:
            for r in csv.reader(f):
                if r[2] in keep and r[1] > last:
                    rows.append((r[1], r[2], r[3]))
    df = pd.DataFrame(rows, columns=["date", "el", "v"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["v"] = pd.to_numeric(df["v"], errors="coerce") / 10.0  # decimi di mm / decimi di °C
    w = df.pivot_table(index="date", columns="el", values="v")
    w["SNOW"] = w["SNOW"] * 10.0  # la neve è in mm interi
    return w.rename(columns=str.lower)


@tidy("cloud_nyc")
def load_cloud(refresh=False):
    """Copertura nuvolosa media 6-16 (ora di NY) a LaGuardia, in ottavi (NOAA ISD-Lite)."""
    out = []
    for year in range(1985, pd.Timestamp.today().year + 1):
        try:
            p = fetch(f"https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/{year}/725030-14732-{year}.gz", f"isd_{year}.gz", refresh and year >= 2025, tries=2)
        except Exception:
            continue
        df = pd.read_csv(p, sep=r"\s+", header=None, compression="gzip", usecols=[0, 1, 2, 3, 9], names=["y", "m", "d", "h", "sky"])
        df = df[(df.h >= 11) & (df.h <= 21) & (df.sky >= 0)]
        df["sky"] = df["sky"].clip(upper=8)
        df["date"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
        g = df.groupby("date")["sky"].agg(["mean", "size"])
        out.append(g[g["size"] >= 3]["mean"])
    return pd.concat(out).rename("cloud_okta")


@tidy("space_weather")
def load_space(refresh=False):
    """Indice geomagnetico Ap e numero di macchie solari (CelesTrak, dal pacchetto `spaceweather`)."""
    import warnings

    import spaceweather as sw

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sw.sw_daily()
    obs = df[(df["Apavg"] >= 0) & (df["Kpsum"] > 0)]
    return obs.loc["1985":, ["Apavg", "isn"]].rename(columns={"Apavg": "ap", "isn": "sunspots"})


# --------------------------------------------------------------------------- calendario FOMC
UNSCHEDULED_AFTER_2018 = {"2019-10-04", "2020-03-03", "2020-03-15", "2020-03-23", "2020-03-31", "2020-04-09"}


@tidy("fomc")
def load_fomc(refresh=False):
    """Giorni di decisione FOMC programmati (1994+). Fino al 2018 dataset accademico, poi calendari pubblici."""
    a = pd.read_csv(fetch(GH + "velikov-mihail/Dambra-Velikov-Weber/HEAD/Inputs/FOMC_dates.csv", "fomc_a.csv", refresh))
    a["date"] = pd.to_datetime(a["date"], format="%m/%d/%Y")
    a = a[(a["scheduled"] == 1) & (a["date"] >= "1994-01-01")]
    b = pd.read_csv(fetch(GH + "CunXin1/qqq_trading/HEAD/datasets/fomc_dates.csv", "fomc_b.csv", refresh), parse_dates=["date"])
    b = b[(b["date"] > a["date"].max()) & (~b["date"].dt.strftime("%Y-%m-%d").isin(UNSCHEDULED_AFTER_2018))]
    dates = pd.DatetimeIndex(sorted(set(a["date"]) | set(b["date"])))
    return pd.DataFrame({"fomc": 1}, index=dates)


LOADERS = {
    "spx": load_spx, "spx_monthly": load_spx_monthly, "fred": load_fred, "hy_oas": load_hy_oas, "vix": load_vix,
    "oil": load_oil, "gold": load_gold, "usd": load_usd, "epu": load_epu, "news_sentiment": load_news_sentiment,
    "gpr": load_gpr, "trump_posts": load_trump, "cot": load_cot, "aaii": load_aaii, "naaim": load_naaim,
    "putcall": load_putcall, "senate": load_senate, "weather_nyc": load_weather, "cloud_nyc": load_cloud,
    "space_weather": load_space, "fomc": load_fomc,
}


def load_all(refresh: bool = False, verbose: bool = True) -> dict[str, pd.DataFrame]:
    data = {}
    for name, fn in LOADERS.items():
        try:
            data[name] = fn(refresh)
            if verbose:
                d = data[name].dropna(how="all")
                print(f"  {name:15s} {len(d):6d} righe  {d.index.min().date()} → {d.index.max().date()}")
        except Exception as e:
            print(f"  {name:15s} NON DISPONIBILE: {e}")
    return data


if __name__ == "__main__":
    import sys

    load_all(refresh="--refresh" in sys.argv)
