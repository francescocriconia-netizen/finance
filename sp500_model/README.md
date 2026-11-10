# Modello di previsione dell'S&P 500 senza prezzi passati

Previsione dell'S&P 500 a 1 giorno, 1 settimana e 1 mese usando **solo informazioni diverse
dallo storico dei prezzi dell'indice**: calendario e ricorrenze, eventi esterni al mercato,
notizie, movimenti di capitale di soggetti specifici, macro, altri mercati.
Il prezzo dell'S&P 500 compare soltanto come variabile da prevedere.

## Come si usa

```bash
pip install -r sp500_model/requirements.txt
python -m sp500_model.run                         # usa i dati salvati in data/tidy
python -m sp500_model.run --refresh               # riscarica tutte le fonti
python -m sp500_model.run --last-price 7549.94 --last-price-date 2026-11-09   # ancora di prezzo più recente
```

Risultati in `sp500_model/output/`: `forecast.json` (previsione), `metrics.json` (validazione),
`ablation.csv` (valore di ogni gruppo), `signals.csv` (catalogo dei singoli segnali),
`oos_predictions.csv.gz`, `equity.csv.gz`, `importance.csv`, `freshness.csv`.

## Cosa guarda il modello (109 variabili, 8 gruppi)

| Gruppo | Esempi | Fonte |
|---|---|---|
| **Calendario di mercato** | fine/inizio mese, scadenze opzioni e "quad witching", pre/post festività, giorni FOMC, venerdì dei dati sul lavoro, stagione delle trimestrali, blackout dei buyback, scadenza fiscale di aprile, ciclo presidenziale, elezioni | calcolato; date FOMC da dataset pubblici |
| **Eventi esterni** | fasi lunari, ore di luce a New York (effetto SAD), lunedì dopo il cambio dell'ora, venerdì 13, Mercurio retrogrado, Mondiali di calcio, Olimpiadi, nuvole/pioggia/neve/temperatura a New York, tempeste geomagnetiche | `ephem`, NOAA GHCN e ISD-Lite, CelesTrak |
| **Notizie** | Economic Policy Uncertainty giornaliero, Daily News Sentiment (Fed di San Francisco), Geopolitical Risk Index, volume e temi dei post di Trump (Twitter 2009-21, Truth Social 2022-25) | mirror pubblici, archivi dei post |
| **Capitali di soggetti specifici** | Fed e Tesoro USA (liquidità netta = bilancio Fed − conto del Tesoro − reverse repo, riserve), asset manager / hedge fund / dealer sui future S&P (CFTC COT), investitori retail (AAII), gestori attivi (NAAIM), opzionisti (put/call), senatori USA (STOCK Act), fondi monetari | FRED, CFTC, AAII, NAAIM, CBOE, Senate Stock Watcher |
| **Macro** | sussidi di disoccupazione, regola di Sahm in tempo reale, fiducia dei consumatori, standard di credito bancario, M2 | FRED |
| **Tassi e credito** | variazioni dei rendimenti a 2 e 10 anni, curva, tassi reali, breakeven, spread high yield e BAA-AAA | FRED, ICE BofA |
| **Altri mercati** | petrolio, dollaro, oro | EIA, Fed H.10 |
| **Volatilità implicita** | VIX: livello, variazioni, percentile a 1 anno | CBOE |

Regole seguite per non "barare":

- ogni dato è usato solo dal giorno in cui era pubblico (ritardi di pubblicazione espliciti in `features.py`: es. COT +6 giorni, bilancio Fed +2, senatori +45);
- le variabili di calendario riguardano i giorni futuri ma sono note in anticipo;
- si usano variazioni e scostamenti dalla media, non livelli con trend;
- serie riviste a posteriori (NFCI, indici di stress) sono escluse.

## Modello e validazione

- Media di **gradient boosting** (LightGBM, alberi molto piccoli) e **ridge**; in più un classificatore per la
  probabilità di rialzo e due regressioni quantiliche per l'intervallo all'80%.
- **Walk-forward**: per ogni anno dal 2005 al 2026 il modello è addestrato solo sui dati precedenti
  (dal 1990), con un embargo pari all'orizzonte. Iperparametri fissati a priori.
- **Selezione dei gruppi senza guardare il futuro**: un gruppo entra nel modello finale solo se, da solo,
  ha funzionato nel periodo di sviluppo 2005-2014. Il periodo **2015-2026 è il banco di prova pulito**.
  Se nessun gruppo passa, la previsione puntuale torna alla media storica.
- Dopo una prima esecuzione che mostrava overfitting a 1 mese ho irrobustito una volta la configurazione
  (alberi più piccoli, niente livelli con trend); da lì in poi nessun ritocco sui dati di verifica.

## Risultati (esecuzione del 10 novembre 2026)

Previsione dalla chiusura del 9 novembre 2026 (7.549,94):

| Orizzonte | Rendimento atteso | Prezzo atteso | Intervallo 80% | Prob. di rialzo |
|---|---|---|---|---|
| 1 giorno | +0,08% | 7.556 | 7.362 – 7.732 | 55% |
| 1 settimana | +0,35% | 7.577 | 7.265 – 7.845 | 62% |
| 1 mese | +0,71% (media storica) | 7.604 | 7.093 – 8.125 | 64% |

Validazione fuori campione (IC = correlazione di rango tra previsione e rendimento realizzato):

| Orizzonte | Gruppi selezionati | IC 2005-2026 | IC verifica 2015-2026 | Nota |
|---|---|---|---|---|
| 1 giorno | calendario + VIX | +0,040 (t 3,0) | +0,035 (t 1,9) | con previsione positiva il giorno dopo ha reso in media +5,5 pb, con previsione negativa +1,5 pb |
| 1 settimana | VIX | +0,067 (t 2,2) | +0,057 (t 1,4) | |
| 1 mese | nessuno | −0,04 | −0,06 | nessuna abilità dimostrata: si usa la media storica |

- Gli intervalli all'80% sono ben calibrati: hanno contenuto il rendimento realizzato nel 79-81% dei casi (76-78% a 1 mese).
- Il segnale c'è ma è piccolo: usato come interruttore "dentro/fuori" (dentro il 74% del tempo, 2015-2026)
  ha reso il 10,3% annuo contro l'11,9% del comprare e tenere, con volatilità 16,3% contro 17,6%.
  Serve quindi a modulare l'esposizione, non a sostituire l'investimento.
- **Modello senza alcun prezzo di mercato** (calendario, eventi esterni, notizie, soggetti, macro): IC ≈ 0
  fuori campione. Presi tutti insieme questi segnali non bastano; presi uno a uno alcuni reggono (sotto).

### Segnali singoli più solidi (campione 1990-2026, coerenti anche dal 2015)

| Segnale | Effetto | t |
|---|---|---|
| VIX alto rispetto all'ultimo anno | rendimenti successivi più alti (IC +0,04 a 1g, +0,09 a 1 sett., +0,12 a 1 mese) | 3,9 / 3,8 / 2,6 |
| Rendimento del 2 anni in salita nell'ultimo mese | rendimenti successivi più bassi (IC −0,03 a 1g, −0,06 a 1 sett.) | −3,0 / −2,6 |
| Giorno di decisione FOMC | +20 pb rispetto agli altri giorni (dal 2015 solo +2 pb) | 2,8 |
| Primo giorno di borsa del mese | +16 pb (dal 2015 +9 pb) | 2,6 |
| Quota di ribassisti AAII alta | rendimenti successivi più alti (contrarian) | 2,4 |
| Liquidità netta Fed/Tesoro in aumento (4 settimane) | rendimenti successivi più alti | 2,1 |
| Stagione delle trimestrali | +5 pb al giorno | 1,9 |
| Mercurio retrogrado | −5 pb al giorno (dal 2015 −11 pb) | −1,7 |
| Neve a New York | −8 pb (dal 2015 −17 pb) | −1,3 |
| Vigilia di festività | +6 pb (dal 2015 +11 pb) | 1,2 |

Con oltre cento segnali testati, quattro o cinque risultano significativi per puro caso: quelli "esoterici"
(Mercurio, neve, venerdì 13) vanno letti così. Luna piena, Mondiali, Olimpiadi e cambio dell'ora non mostrano
effetti misurabili. L'elenco completo è in `output/signals.csv`.

## Limiti e prossimi passi

- **Fonti**: l'ambiente in cui è stato costruito il modello raggiunge solo GitHub e gli open data NOAA su AWS.
  Molte serie arrivano quindi da mirror pubblici di FRED/CFTC/AAII ecc. (elenco in `sources.py`), verificati dove
  possibile (lo storico S&P coincide con FRED allo 0,0015%). Con accesso diretto a FRED, SEC, CFTC e Tesoro
  si possono aggiungere: acquisti/vendite degli insider (Form 4), portafogli 13F, operazioni dei parlamentari
  aggiornate, Daily Treasury Statement, GDELT, visualizzazioni Wikipedia, Google Trends, flussi ETF.
- **Dati non aggiornati alla data della previsione** (usati nello storico, assenti oggi): EPU (fino a set 2026),
  news sentiment e GPR (giu 2026), NAAIM (lug 2026), post di Trump (ott 2025), nuvolosità (ago 2025), senatori (2020).
  Il modello finale selezionato usa solo calendario e VIX, entrambi aggiornati.
- Lo storico dei prezzi nel repository arriva al 21 ottobre 2026; il livello di partenza della previsione
  (7.549,94 del 9 novembre) è passato a mano con `--last-price`.
- Costi di transazione considerati: 1 pb per cambio di posizione; la liquidità rende zero.

Non è una consulenza finanziaria: è uno strumento statistico con un margine piccolo e misurato.
