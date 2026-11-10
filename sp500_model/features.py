"""Variabili esplicative del modello, organizzate per gruppi.

Regola di base: ogni variabile alla data t usa solo informazioni già pubbliche alla
chiusura di t (le serie sono spostate in avanti del loro ritardo di pubblicazione).
Le variabili di calendario riguardano invece la finestra futura t+1..t+h: sono
deterministiche e note in anticipo, quindi non introducono look-ahead.

Nessuna variabile deriva dallo storico dei prezzi dell'S&P 500.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMemorialDay,
                                    USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday)
from dateutil.relativedelta import MO
from pandas.tseries.offsets import DateOffset

GROUPS = {
    "cal": "Calendario di mercato (fine mese, scadenze opzioni, festività, FOMC, trimestrali, elezioni)",
    "ext": "Eventi esterni (luna, luce diurna, ora legale, Mercurio retrogrado, sport, meteo di New York, sole)",
    "news": "Notizie (incertezza politica, tono delle news, rischio geopolitico, post di Trump)",
    "sog": "Capitali di soggetti specifici (Fed, Tesoro, asset manager, hedge fund, dealer, retail, senatori)",
    "mac": "Macro (sussidi di disoccupazione, fiducia consumatori, credito bancario, moneta)",
    "tc": "Tassi e credito (rendimenti, curva, tassi reali, spread high yield)",
    "xa": "Altri mercati (petrolio, dollaro, oro)",
    "vix": "Volatilità implicita (VIX)",
}
NO_MARKET_GROUPS = ["cal", "ext", "news", "sog", "mac"]  # variante "senza prezzi di mercato"


def group_of(col: str) -> str:
    return col.split("_", 1)[0]


# --------------------------------------------------------------------------- calendario di borsa
class NYSECalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("Capodanno", month=1, day=1, observance=sunday_to_monday),
        Holiday("MLK", month=1, day=1, offset=DateOffset(weekday=MO(3)), start_date="1998-01-01"),
        USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-01", observance=nearest_workday),
        Holiday("4 luglio", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Natale", month=12, day=25, observance=nearest_workday),
    ]


def trading_index(spx_index: pd.DatetimeIndex, until: pd.Timestamp) -> pd.DatetimeIndex:
    """Giorni di borsa: quelli storici dell'S&P più quelli futuri generati dal calendario NYSE."""
    last = spx_index[-1]
    end = (until + pd.offsets.MonthEnd(0)) if until > last else last
    fut = pd.bdate_range(last + pd.Timedelta(days=1), end)
    fut = fut.difference(NYSECalendar().holidays(last, end))
    return spx_index.append(fut)


# --------------------------------------------------------------------------- astronomia
def _new_moons(start="1984-01-01", end="2030-01-01") -> pd.DatetimeIndex:
    import ephem

    out, d = [], ephem.Date(start.replace("-", "/"))
    stop = ephem.Date(end.replace("-", "/"))
    while d < stop:
        d = ephem.next_new_moon(d)
        out.append(pd.Timestamp(ephem.Date(d).datetime()))
        d = ephem.Date(d + 1)
    return pd.DatetimeIndex(out)


def moon_age(dates: pd.DatetimeIndex) -> np.ndarray:
    """Giorni trascorsi dall'ultima luna nuova (alle 16:00 UTC, apertura di Wall Street)."""
    nm = _new_moons()
    t = dates + pd.Timedelta(hours=16)
    pos = np.searchsorted(nm.values, t.values, side="right") - 1
    return (t.values - nm.values[pos]) / np.timedelta64(1, "D")


def mercury_retrograde(dates: pd.DatetimeIndex) -> np.ndarray:
    """1 se Mercurio è in moto retrogrado apparente (longitudine eclittica geocentrica in calo)."""
    import ephem

    days = pd.date_range(dates.min() - pd.Timedelta(days=2), dates.max() + pd.Timedelta(days=1), freq="D")
    m = ephem.Mercury()
    lon = np.empty(len(days))
    for i, d in enumerate(days):
        m.compute(ephem.Date(d.to_pydatetime()))
        lon[i] = float(ephem.Ecliptic(m).lon)
    dlon = np.diff(np.unwrap(lon))
    retro = pd.Series(np.r_[0.0, (dlon < 0).astype(float)], index=days)
    return retro.reindex(dates).values


def daylight_hours(dates: pd.DatetimeIndex, lat: float = 40.71) -> np.ndarray:
    """Ore di luce a New York (formula astronomica)."""
    n = dates.dayofyear.values
    decl = -23.44 * np.cos(2 * np.pi * (n + 10) / 365.25)
    x = -np.tan(np.radians(lat)) * np.tan(np.radians(decl))
    return 24 / np.pi * np.arccos(np.clip(x, -1, 1))


def dst_switch_days(years) -> pd.DatetimeIndex:
    """Domeniche di cambio dell'ora negli USA (regole 1987-2006 e dal 2007)."""
    out = []
    for y in years:
        if y >= 2007:
            a = pd.Timestamp(y, 3, 1) + pd.offsets.Week(weekday=6) * 2 if pd.Timestamp(y, 3, 1).weekday() != 6 else pd.Timestamp(y, 3, 8)
            b = pd.Timestamp(y, 11, 1) + pd.offsets.Week(weekday=6) if pd.Timestamp(y, 11, 1).weekday() != 6 else pd.Timestamp(y, 11, 1)
        else:
            a = pd.Timestamp(y, 4, 1) + pd.offsets.Week(weekday=6) if pd.Timestamp(y, 4, 1).weekday() != 6 else pd.Timestamp(y, 4, 1)
            b = pd.Timestamp(y, 10, 31) - pd.offsets.Week(weekday=6) if pd.Timestamp(y, 10, 31).weekday() != 6 else pd.Timestamp(y, 10, 31)
        out += [a, b]
    return pd.DatetimeIndex(out)


WORLD_CUPS = [("1990-06-08", "1990-07-08"), ("1994-06-17", "1994-07-17"), ("1998-06-10", "1998-07-12"),
              ("2002-05-31", "2002-06-30"), ("2006-06-09", "2006-07-09"), ("2010-06-11", "2010-07-11"),
              ("2014-06-12", "2014-07-13"), ("2018-06-14", "2018-07-15"), ("2022-11-20", "2022-12-18"),
              ("2026-06-11", "2026-07-19")]
OLYMPICS = [("1992-07-25", "1992-08-09"), ("1996-07-19", "1996-08-04"), ("2000-09-15", "2000-10-01"),
            ("2004-08-13", "2004-08-29"), ("2008-08-08", "2008-08-24"), ("2012-07-27", "2012-08-12"),
            ("2016-08-05", "2016-08-21"), ("2021-07-23", "2021-08-08"), ("2024-07-26", "2024-08-11")]


def _in_windows(idx, windows) -> np.ndarray:
    out = np.zeros(len(idx), dtype=float)
    for a, b in windows:
        out[(idx >= a) & (idx <= b)] = 1.0
    return out


def _election_day(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year, 11, 1)
    first_monday = d + pd.Timedelta(days=(7 - d.weekday()) % 7)
    return first_monday + pd.Timedelta(days=1)


# --------------------------------------------------------------------------- flag giornalieri di calendario
def daily_flags(idx: pd.DatetimeIndex, fomc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Flag 0/1 per ogni giorno di borsa (tutti noti in anticipo)."""
    f = pd.DataFrame(index=idx)
    ym = idx.to_period("M")
    pos = pd.Series(np.arange(len(idx)), index=idx)
    k_start = pos.groupby(ym).cumcount().values
    k_end = pos.groupby(ym).transform("size").values - 1 - k_start
    month, day, wd = idx.month.values, idx.day.values, idx.weekday.values

    f["cal_tom"] = ((k_end == 0) | (k_start <= 2)).astype(float)            # turn of the month
    f["cal_month_first"] = (k_start == 0).astype(float)
    f["cal_qend"] = (np.isin(month, [3, 6, 9, 12]) & (k_end <= 2)).astype(float)  # ribilanciamenti di fine trimestre
    f["cal_santa"] = (((month == 12) & (k_end <= 4)) | ((month == 1) & (k_start <= 1))).astype(float)

    first = ym.to_timestamp()
    third_fri = first + pd.to_timedelta(((4 - first.weekday.values) % 7) + 14, unit="D")
    d_opex = (third_fri - idx).days.values
    f["cal_opex_week"] = ((d_opex >= 0) & (d_opex <= 4)).astype(float)
    f["cal_post_opex_week"] = ((d_opex <= -3) & (d_opex >= -7)).astype(float)
    f["cal_quad_week"] = (f["cal_opex_week"].values * np.isin(month, [3, 6, 9, 12])).astype(float)

    hol = NYSECalendar().holidays(idx.min() - pd.Timedelta(days=10), idx.max() + pd.Timedelta(days=10))
    nxt_bd, prv_bd = idx + pd.offsets.BDay(1), idx - pd.offsets.BDay(1)
    f["cal_pre_holiday"] = nxt_bd.isin(hol).astype(float)
    f["cal_post_holiday"] = prv_bd.isin(hol).astype(float)

    f["cal_fomc"] = idx.isin(fomc_dates).astype(float)
    f["cal_jobs_friday"] = ((wd == 4) & (day <= 7)).astype(float)

    q_end = (idx - pd.offsets.QuarterEnd(1)).where(~idx.is_quarter_end, idx)  # ultimo fine trimestre <= d
    dq = (idx - q_end).days.values
    to_q = ((idx + pd.offsets.QuarterEnd(0)) - idx).days.values
    f["cal_earnings_season"] = ((dq >= 12) & (dq <= 45)).astype(float)
    f["cal_buyback_blackout"] = ((to_q <= 14) | (dq <= 21)).astype(float)
    f["cal_tax_april"] = ((month == 4) & (day >= 8) & (day <= 18)).astype(float)

    years = range(idx.year.min() - 1, idx.year.max() + 2)
    pre_el = np.zeros(len(idx))
    post_el = np.zeros(len(idx))
    for y in years:
        if y % 2 == 0:
            e = _election_day(y)
            pre_el[(idx > e - pd.Timedelta(days=45)) & (idx <= e)] = 1
            post_el[(idx > e) & (idx <= e + pd.Timedelta(days=30))] = 1
    f["cal_pre_election"], f["cal_post_election"] = pre_el, post_el

    # --- eventi "esterni" al mercato
    sw = dst_switch_days(years)
    first_after = idx[np.clip(np.searchsorted(idx.values, sw.values, side="right"), 0, len(idx) - 1)]
    f["ext_dst_monday"] = idx.isin(first_after).astype(float)
    f["ext_friday13"] = ((wd == 4) & (day == 13)).astype(float)
    f["ext_world_cup"] = _in_windows(idx, WORLD_CUPS)
    f["ext_olympics"] = _in_windows(idx, OLYMPICS)
    age = moon_age(idx)
    f["ext_full_moon"] = (np.abs(age - 14.765) <= 3.5).astype(float)
    f["ext_new_moon"] = ((age <= 3.5) | (age >= 26.0)).astype(float)
    f["ext_mercury_retro"] = mercury_retrograde(idx)
    return f


def calendar_features(idx: pd.DatetimeIndex, fomc_dates: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    """Variabili di calendario per la finestra futura t+1..t+h (quota di giorni con il flag attivo)."""
    flags = daily_flags(idx, fomc_dates)
    out = flags.shift(-1).rolling(h).mean().shift(-(h - 1))  # media su t+1..t+h
    nxt = pd.Series(idx, index=idx).shift(-1)                # prossimo giorno di borsa
    nd = pd.DatetimeIndex(nxt.fillna(idx[-1]))
    if h == 1:
        out["cal_monday"] = (nd.weekday == 0).astype(float)
        out["cal_friday"] = (nd.weekday == 4).astype(float)
    out["cal_month_sin"] = np.sin(2 * np.pi * (nd.dayofyear.values / 365.25))
    out["cal_month_cos"] = np.cos(2 * np.pi * (nd.dayofyear.values / 365.25))
    out["cal_halloween"] = np.isin(nd.month, [11, 12, 1, 2, 3, 4]).astype(float)   # "sell in May"
    out["cal_pres_year"] = ((nd.year.values - 1) % 4 + 1).astype(float)             # anno del ciclo presidenziale
    # giorni di borsa alla prossima decisione FOMC
    fpos = np.searchsorted(idx.values, fomc_dates.values)
    nxt_f = np.searchsorted(fpos, np.arange(len(idx)), side="right")
    dist = np.where(nxt_f < len(fpos), fpos[np.clip(nxt_f, 0, len(fpos) - 1)] - np.arange(len(idx)), np.nan)
    dist[idx < "1994-01-01"] = np.nan
    out["cal_days_to_fomc"] = np.clip(dist, 0, 40)
    # luce e luna alla data del prossimo giorno di borsa
    dl = daylight_hours(nd)
    doy = nd.dayofyear.values
    fall_winter = (doy >= 264) | (doy <= 79)
    out["ext_daylight"] = dl
    out["ext_sad_night"] = np.where(fall_winter, 12.0 - dl, 0.0)             # Kamstra-Kramer-Levi
    out["ext_fall"] = ((doy >= 264) & (doy <= 354)).astype(float)
    out["ext_moon_cos"] = np.cos(2 * np.pi * moon_age(nd) / 29.530588)       # +1 luna nuova, -1 luna piena
    return out


# --------------------------------------------------------------------------- allineamento "point in time"
def avail(s: pd.Series, idx: pd.DatetimeIndex, lag: int = 0, stale: int | None = None) -> pd.Series:
    """Sposta `s` di `lag` giorni di calendario (ritardo di pubblicazione) e la riporta sui giorni di borsa.

    `stale`: età massima (giorni) oltre la quale il dato è considerato scaduto e diventa NaN.
    """
    s = s.dropna()
    a = pd.Series(s.values, index=s.index + pd.Timedelta(days=lag))
    a = a[~a.index.duplicated(keep="last")]
    u = a.index.union(idx)
    out = a.reindex(u).ffill().reindex(idx)
    if stale is not None:
        stamp = pd.Series(a.index, index=a.index).reindex(u).ffill().reindex(idx)
        age = (idx - pd.DatetimeIndex(stamp)).days
        out[np.asarray(age > stale) | stamp.isna().values] = np.nan
    return out


def _z(x: pd.Series, win: int) -> pd.Series:
    return (x - x.rolling(win, min_periods=win // 2).mean()) / x.rolling(win, min_periods=win // 2).std()


def state_features(data: dict, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Variabili di stato note alla chiusura di t (con ritardi di pubblicazione).

    Si usano variazioni e scostamenti dalla propria media, non livelli con trend: livelli che
    cambiano lentamente finirebbero per fare da "orologio" e far memorizzare al modello i singoli periodi.
    """
    F: dict[str, pd.Series] = {}
    fr = data["fred"]

    # ---------------- soggetti: Fed e Tesoro (bilancio del mercoledì, pubblico dal giovedì sera -> lag 2)
    walcl, tga, res = fr["WALCL"].dropna(), fr["WTREGEN"].dropna(), fr["WRESBAL"].dropna()
    rrp = fr["RRPONTSYD"].dropna()
    rrp_w = rrp.reindex(walcl.index.union(rrp.index)).ffill().reindex(walcl.index).fillna(0) * 1000.0  # mld -> mln
    tga_w, res_w = tga.reindex(walcl.index), res.reindex(walcl.index)
    netliq = walcl - tga_w - rrp_w                                  # "liquidità netta" = bilancio Fed - conto Tesoro - reverse repo
    F["sog_netliq_chg4w"] = avail(netliq.diff(4) / walcl, idx, 2, 21)
    F["sog_netliq_chg13w"] = avail(netliq.diff(13) / walcl, idx, 2, 21)
    F["sog_fed_assets_chg13w"] = avail(np.log(walcl).diff(13), idx, 2, 21)
    F["sog_treasury_cash_chg4w"] = avail(tga_w.diff(4) / walcl, idx, 2, 21)
    F["sog_reserves_chg4w"] = avail(res_w.diff(4) / walcl, idx, 2, 21)
    F["sog_rrp_chg4w"] = avail(rrp_w.diff(4) / walcl, idx, 2, 21)

    # ---------------- soggetti: posizionamento sui future (martedì, pubblicato venerdì -> lag 6 prudenziale)
    c = data["cot"]
    for col, nm in [("asset_mgr_net", "asset_mgr"), ("lev_money_net", "hedge_funds"), ("dealer_net", "dealers")]:
        x = c[col] / c["open_interest"]
        F[f"sog_cot_{nm}_chg4w"] = avail(x.diff(4), idx, 6, 21)
        F[f"sog_cot_{nm}_z52"] = avail(_z(x, 52), idx, 6, 21)

    # ---------------- soggetti: investitori retail (AAII) e gestori attivi (NAAIM) -> lag 2
    a = data["aaii"]
    spread = a["bullish"] - a["bearish"]
    F["sog_aaii_spread"] = avail(spread, idx, 2, 21)
    F["sog_aaii_spread_8w"] = avail(spread.rolling(8).mean(), idx, 2, 21)
    F["sog_aaii_bearish"] = avail(a["bearish"], idx, 2, 21)
    n = data["naaim"]["naaim"]
    F["sog_naaim"] = avail(n, idx, 2, 21)
    F["sog_naaim_chg4w"] = avail(n.diff(4), idx, 2, 21)

    # ---------------- soggetti: opzionisti (put/call sulle azioni, pubblicato a mercati chiusi -> lag 1)
    pc = data["putcall"]["equity_pc"]
    pc = pc[pc > 0]
    F["sog_putcall_5d_z"] = avail(_z(pc.rolling(5).mean(), 252), idx, 1, 7)
    F["sog_putcall_21d_z"] = avail(_z(pc.rolling(21).mean(), 252), idx, 1, 7)

    # ---------------- soggetti: senatori USA (dichiarazione entro 45 giorni -> lag 45)
    s = data["senate"]
    b90, s90 = s["n_buy"].rolling(90).sum(), s["n_sell"].rolling(90).sum()
    tot = b90 + s90
    F["sog_senators_netbuy90"] = avail(((b90 - s90) / tot).where(tot >= 10), idx, 45, 10)
    F["sog_senators_activity90"] = avail(np.log1p(tot), idx, 45, 10)

    # ---------------- soggetti: fondi monetari (trimestrale, pubblicato ~75 giorni dopo fine trimestre)
    mmf = fr["MMMFFAQ027S"].dropna()
    F["sog_mmf_assets_yoy"] = avail(np.log(mmf).diff(4), idx, 165, 200)

    # ---------------- notizie
    e = data["epu"]["epu"].clip(lower=1)
    F["news_epu_30d_vs_1y"] = avail(np.log(e.rolling(30).mean() / e.rolling(365).mean()), idx, 1, 7)
    F["news_epu_vs_1y"] = avail(np.log(e.rolling(5).mean() / e.rolling(365).mean()), idx, 1, 7)
    F["news_epu_spike"] = avail(np.log(e / e.rolling(30).mean()), idx, 1, 7)
    ns = data["news_sentiment"]["news_sentiment"]
    F["news_sentiment"] = avail(ns, idx, 7, 14)
    F["news_sentiment_chg30"] = avail(ns.diff(30), idx, 7, 14)
    F["news_sentiment_chg90"] = avail(ns.diff(90), idx, 7, 14)
    g = data["gpr"]["gpr"].clip(lower=1)
    F["news_gpr_7d_vs_1y"] = avail(np.log(g.rolling(7).mean() / g.rolling(365).mean()), idx, 2, 10)
    F["news_gpr_30d_vs_1y"] = avail(np.log(g.rolling(30).mean() / g.rolling(365).mean()), idx, 2, 10)
    t = data["trump_posts"]
    F["news_trump_surge"] = avail(np.log1p(t["posts"].rolling(5).mean()) - np.log1p(t["posts"].rolling(90).mean()), idx, 1, 7)
    F["news_trump_market_posts5"] = avail(np.log1p((t["mkt"] + t["fed"]).rolling(5).sum()), idx, 1, 7)
    F["news_trump_trade_posts5"] = avail(np.log1p(t["trade"].rolling(5).sum()), idx, 1, 7)

    # ---------------- macro (ritardi di pubblicazione prudenziali)
    ic = fr["ICSA"].dropna()
    a4 = ic.rolling(4).mean()
    F["mac_claims_trend"] = avail(np.log(a4 / ic.rolling(26).mean()), idx, 5, 21)
    F["mac_claims_yoy"] = avail(np.log(a4 / a4.shift(52)), idx, 5, 21)
    F["mac_sahm_rule"] = avail(fr["SAHMREALTIME"].dropna(), idx, 40, 75)
    um = fr["UMCSENT"].dropna()
    F["mac_consumer_sent_z"] = avail(_z(um, 60), idx, 30, 75)
    F["mac_consumer_sent_chg3m"] = avail(um.diff(3), idx, 30, 75)
    F["mac_lending_standards"] = avail(fr["DRTSCILM"].dropna(), idx, 45, 150)
    m2 = fr["WM2NS"].dropna()
    F["mac_m2_yoy"] = avail(np.log(m2).diff(52), idx, 28, 75)

    # ---------------- tassi e credito (prezzi di altri mercati, osservabili in giornata)
    y10, y2, y3m = (avail(fr[k], idx, 0, 7) for k in ("DGS10", "DGS2", "DGS3MO"))
    F["tc_y10_chg5"], F["tc_y10_chg21"], F["tc_y10_chg63"] = y10.diff(5), y10.diff(21), y10.diff(63)
    F["tc_y2_chg21"] = y2.diff(21)
    F["tc_y3m_chg63"] = y3m.diff(63)
    F["tc_curve_10y2y"] = y10 - y2
    F["tc_curve_10y3m"] = y10 - y3m
    F["tc_curve_chg63"] = (y10 - y2).diff(63)
    real, be = avail(fr["DFII10"], idx, 0, 7), avail(fr["T10YIE"], idx, 0, 7)
    F["tc_real10_chg21"], F["tc_breakeven_chg21"] = real.diff(21), be.diff(21)
    hy = avail(data["hy_oas"]["hy_oas"], idx, 1, 7)
    F["tc_hy_spread"] = hy
    F["tc_hy_spread_chg21"] = hy.diff(21)
    F["tc_hy_spread_z252"] = _z(hy, 252)
    baa, aaa = avail(fr["DBAA"], idx, 1, 7), avail(fr["DAAA"], idx, 1, 7)
    F["tc_baa_aaa"] = baa - aaa
    F["tc_baa_aaa_chg21"] = (baa - aaa).diff(21)

    # ---------------- altri mercati
    oil = np.log(avail(data["oil"]["wti"].clip(lower=1), idx, 0, 7))
    usd = avail(data["usd"]["usd_log"], idx, 0, 7)
    for k in (5, 21, 63):
        F[f"xa_oil_ret{k}"] = oil.diff(k)
        F[f"xa_usd_ret{k}"] = usd.diff(k)
    F["xa_oil_vs_1y"] = oil - oil.rolling(252, min_periods=120).mean()
    gold = data["gold"]["gold"].dropna()
    F["xa_gold_ret3m"] = avail(np.log(gold).diff(3), idx, 31, 75)

    # ---------------- volatilità implicita
    vix = np.log(avail(data["vix"]["vix"], idx, 0, 7))
    F["vix_level"] = vix
    F["vix_chg5"], F["vix_chg21"] = vix.diff(5), vix.diff(21)
    F["vix_vs_3m"] = vix - vix.rolling(63, min_periods=30).mean()
    F["vix_pct_1y"] = vix.rolling(252, min_periods=120).rank(pct=True)

    # ---------------- natura: meteo di New York (osservabile in giornata)
    cl = data["cloud_nyc"]["cloud_okta"]
    clim = cl[:"1999"].groupby(cl[:"1999"].index.month).mean()
    cl_an = cl - cl.index.month.map(clim)
    F["ext_cloud_anomaly"] = avail(cl_an, idx, 0, 3)
    F["ext_cloud_anomaly_5d"] = avail(cl_an.rolling(5).mean(), idx, 0, 3)
    w = data["weather_nyc"]
    tavg = (w["tmax"] + w["tmin"]) / 2
    tclim = tavg[:"1999"].groupby(tavg[:"1999"].index.month).mean()
    F["ext_temp_anomaly"] = avail(tavg - tavg.index.month.map(tclim), idx, 0, 5)
    F["ext_rain"] = avail(np.log1p(w["prcp"].clip(lower=0)), idx, 0, 5)
    F["ext_snow"] = avail((w["snow"] > 0).astype(float).where(w["snow"].notna()), idx, 0, 5)

    # ---------------- natura: attività solare
    sp = data["space_weather"]
    ap6 = sp["ap"].rolling(6).max()
    F["ext_geomagnetic_6d"] = avail(np.log1p(ap6), idx, 1, 5)
    F["ext_geomagnetic_storm"] = avail((ap6 >= 40).astype(float), idx, 1, 5)

    return pd.DataFrame(F, index=idx)


# --------------------------------------------------------------------------- dataset finale
def build_dataset(data: dict, horizons=(1, 5, 21), start="1990-01-02", today: pd.Timestamp | None = None):
    """Restituisce {h: X_h}, y (rendimenti logaritmici futuri), prezzi e data di riferimento della previsione."""
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    px = data["spx"]["close"].dropna()
    idx = trading_index(px.index, today + pd.Timedelta(days=75))
    fomc = data["fomc"].index
    state = state_features(data, idx)
    asof = idx[idx < today][-1]            # ultima chiusura già avvenuta
    keep = (idx >= start) & (idx <= asof)
    X, Y = {}, {}
    lp = np.log(px.reindex(idx))
    for h in horizons:
        cal = calendar_features(idx, fomc, h)
        X[h] = pd.concat([cal, state], axis=1).loc[keep]
        Y[h] = (lp.shift(-h) - lp).loc[keep].rename(f"ret_{h}d")
    return X, pd.DataFrame(Y), px, asof
