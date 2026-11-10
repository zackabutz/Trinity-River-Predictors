# Headwater rain station comparison

Period: 2005-01-01 through 2025-12-31. Target: next calendar day's USGS daily
mean flow. Baseline: the project's existing flow and local-weather features.
Each added scenario supplies the named headwater station's precipitation dated
no later than the feature date. Models were trained on target dates through
2019-09-22, validated through 2022-11-11, and tested after that date. All
scenarios used identical target rows and splits.

## Candidate stations

| Fork | NOAA station | Distance from approximate source | Quality-accepted precipitation days, 2005–2025 |
|---|---|---:|---:|
| West | [Antelope, USC00410271](https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USC00410271.csv.gz) | 24.6 km | 7,453 / 7,670 (97.2%) |
| Clear | [Mineral Wells AP, USW00093985](https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00093985.csv.gz) | 26.1 km | 7,643 / 7,670 (99.6%) |
| Elm | [Gainesville 5 ENE, USC00413420](https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USC00413420.csv.gz) | 45.8 km | 7,593 / 7,670 (99.0%) |
| East | [Sherman, USC00418274](https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USC00418274.csv.gz) | 19.3 km | 7,598 / 7,670 (99.1%) |

Elm's nearest long-running precipitation station, Bonita 7NW, is 21.5 km from
the approximate source but has only 80.6% precipitation coverage for this
period. Gainesville was selected to reduce missingness; it is a less local
proxy for the source. The distances are geographic, not watershed distances.

The source locations come from the Handbook of Texas entries for the
[West](https://www.tshaonline.org/handbook/entries/west-fork-of-the-trinity-river),
[Clear](https://www.tshaonline.org/handbook/entries/clear-fork-of-the-trinity-river),
[Elm](https://www.tshaonline.org/handbook/entries/elm-fork-of-the-trinity-river),
and [East](https://www.tshaonline.org/handbook/entries/east-fork-of-the-trinity-river)
forks. Coverage counts come from downloaded NOAA GHCN-Daily station records
after excluding precipitation observations with nonblank quality flags.

## All four stations versus baseline

Mean absolute error (MAE), in cfs; lower is better.

| USGS gage | Model | Validation baseline | Validation + all | Test baseline | Test + all |
|---|---|---:|---:|---:|---:|
| Rosser | Linear regression | 873 | 848 | 1,053 | 1,039 |
| Rosser | Gradient boosting | 859 | 774 | 1,020 | 979 |
| Trinidad | Linear regression | 877 | 848 | 988 | 970 |
| Trinidad | Gradient boosting | 881 | 836 | 1,018 | 966 |
| Goodrich | Linear regression | 661 | 718 | 1,120 | 1,157 |
| Goodrich | Gradient boosting | 682 | 716 | 1,547 | 1,553 |
| All three | Linear regression | 804 | 805 | 1,054 | 1,056 |
| All three | Gradient boosting | 807 | 776 | 1,195 | 1,166 |

The combined rainfall features improved gradient boosting at Rosser and
Trinidad in both validation and test periods. They did not improve Goodrich;
the upstream headwater gauges are much farther from that gage and do not
describe the intervening basin's rainfall. Linear regression had mixed
results. The individual-station MAE changes are in
`data/headwater_comparison/metrics.csv`. These are modest retrospective gains,
not evidence of improved real-time forecasts: NOAA daily reports may arrive
after the observation date, and an observer's 24-hour precipitation period may
not match the USGS calendar day.

Reproduce with `.venv/bin/python compare_headwater_weather.py` from this folder.
The script writes station coverage, metrics, and predictions into
`data/headwater_comparison/`.
