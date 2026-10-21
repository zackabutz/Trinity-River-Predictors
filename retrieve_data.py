#!/usr/bin/env python3
"""Retrieve daily Trinity River discharge and nearby GHCN-Daily weather.

Uses the USGS OGC Water Data API and NOAA's public GHCN-Daily station files.
Raw downloads are cached so a failed run can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

USGS = "https://api.waterdata.usgs.gov/ogcapi/v1/collections"
GHCN = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
SITES = {"Rosser": "08062500", "Trinidad": "08062700", "Goodrich": "08066250"}
ELEMENTS = {"PRCP": ("prcp_mm", 0.1), "TMAX": ("tmax_c", 0.1),
            "TMIN": ("tmin_c", 0.1), "SNOW": ("snow_mm", 1.0),
            "SNWD": ("snwd_mm", 1.0), "ASLP": ("pressure_hpa", 0.1)}
HEADERS = {"User-Agent": "trinity-streamflow-course-project/1.0", "Accept": "application/json,text/plain"}


def fetch(url: str, params: dict | None = None, attempts: int = 4) -> bytes:
    if params:
        url += "?" + urlencode(params)
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=90) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise RuntimeError(f"HTTP {exc.code}: {url}") from exc
        except URLError as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def cached(path: Path, url: str, refresh: bool = False) -> bytes:
    if refresh or not path.exists():
        data = fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.read_bytes()


def json_pages(url: str, params: dict):
    visited = set()
    while url:
        full_url = url + ("?" + urlencode(params) if params else "")
        if full_url in visited:
            raise RuntimeError(f"Repeated pagination URL: {full_url}")
        visited.add(full_url)
        page = json.loads(fetch(url, params))
        yield page
        url = next((link.get("href") for link in page.get("links", [])
                    if link.get("rel") == "next"), None)
        params = {}


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_site(name: str, number: str) -> dict:
    url = f"{USGS}/monitoring-locations/items/USGS-{number}"
    feature = json.loads(fetch(url, {"f": "json"}))
    props = feature["properties"]
    lon, lat = feature["geometry"]["coordinates"][:2]
    return {"site_name": name, "site_number": number, "monitoring_location_id": f"USGS-{number}",
            "latitude": lat, "longitude": lon,
            "drainage_area_sqmi": props.get("drainage_area"),
            "elevation_ft": props.get("altitude"), "huc": props.get("hydrologic_unit_code")}


def get_flow(site: dict, start: date, end: date) -> list[dict]:
    params = {"f": "json", "monitoring_location_id": site["monitoring_location_id"],
              "parameter_code": "00060", "statistic_id": "00003",
              "datetime": f"{start}/{end}", "limit": 10000}
    rows = {}
    for page in json_pages(f"{USGS}/daily/items", params):
        for feature in page.get("features", []):
            prop = feature["properties"]
            day = str(prop.get("time", ""))[:10]
            if not day or prop.get("value") in (None, ""):
                continue
            try:
                value = float(prop["value"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value < 0:
                continue
            unit = prop.get("unit_of_measure")
            if unit not in ("ft^3/s", "ft3/s", "cfs"):
                raise ValueError(f"Unexpected flow unit {unit!r} at {site['site_number']}; expected cfs")
            record = {"site_number": site["site_number"], "date": day, "flow_cfs": value,
                      "approval_status": prop.get("approval_status", ""),
                      "qualifier": prop.get("qualifier", "")}
            # One daily mean per gage/date; prefer approved if duplicate series exist.
            if day not in rows or (record["approval_status"] == "Approved" and rows[day]["approval_status"] != "Approved"):
                rows[day] = record
    return [rows[day] for day in sorted(rows)]


def ghcn_stations(raw: bytes) -> list[dict]:
    stations = []
    for line in raw.decode("ascii", errors="replace").splitlines():
        try:
            stations.append({"station_id": line[:11].strip(), "latitude": float(line[12:20]),
                             "longitude": float(line[21:30]), "station_name": line[41:71].strip()})
        except ValueError:
            continue
    return stations


def eligible_station_ids(raw: bytes, start: date, end: date, min_coverage: float) -> set[str]:
    """Use NOAA's element inventory to exclude stations with inadequate periods.

    The inventory records first/last years, not daily completeness. Actual
    coverage is checked after downloading each candidate station file.
    """
    required = {"PRCP", "TMAX", "TMIN"}
    spans: dict[str, dict[str, tuple[int, int]]] = {}
    for line in raw.decode("ascii", errors="replace").splitlines():
        element = line[31:35].strip()
        if element not in required:
            continue
        try:
            first, last = int(line[36:40]), int(line[41:45])
        except ValueError:
            continue
        spans.setdefault(line[:11].strip(), {})[element] = (first, last)
    years = end.year - start.year + 1
    eligible = set()
    for sid, elements in spans.items():
        if not required.issubset(elements):
            continue
        if all((min(end.year, elements[key][1]) - max(start.year, elements[key][0]) + 1) / years
               >= min_coverage for key in required):
            eligible.add(sid)
    return eligible


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def station_weather(raw: bytes, start: date, end: date) -> list[dict]:
    days: dict[str, dict] = {}
    for row in csv.reader(io.StringIO(raw.decode("utf-8-sig"))):
        if len(row) < 6 or row[2] not in ELEMENTS or row[5].strip():
            continue  # NOAA quality flag: nonblank means the value failed checks.
        try:
            day = date.fromisoformat(f"{row[1][:4]}-{row[1][4:6]}-{row[1][6:8]}")
            value = int(row[3])
        except ValueError:
            continue
        if not start <= day <= end or value == -9999:
            continue
        column, factor = ELEMENTS[row[2]]
        days.setdefault(day.isoformat(), {"date": day.isoformat()})[column] = value * factor
    return [days[day] for day in sorted(days)]


def choose_weather(site: dict, stations: list[dict], eligible: set[str], raw_dir: Path, start: date, end: date,
                   radius_km: float, candidates: int, min_coverage: float, refresh: bool):
    nearby = []
    for station in stations:
        if station["station_id"] not in eligible:
            continue
        km = distance_km(site["latitude"], site["longitude"], station["latitude"], station["longitude"])
        if km <= radius_km:
            nearby.append((km, station))
    nearby.sort(key=lambda x: x[0])
    total_days = (end - start).days + 1
    best = None
    for km, station in nearby[:candidates]:
        sid = station["station_id"]
        try:
            compressed = cached(raw_dir / "noaa" / f"{sid}.csv.gz",
                                f"{GHCN}/by_station/{sid}.csv.gz", refresh)
            raw = gzip.decompress(compressed)
            weather = station_weather(raw, start, end)
        except (RuntimeError, OSError) as exc:
            print(f"Skipping {sid}: {exc}", file=sys.stderr)
            continue
        available = {row["date"] for row in weather if all(k in row for k in ("prcp_mm", "tmax_c", "tmin_c"))}
        coverage = len(available) / total_days
        candidate = (coverage, -km, station, weather)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        if coverage >= min_coverage:
            break  # nearest station with adequate complete daily weather
    if best is None or best[0] < min_coverage:
        raise RuntimeError(f"No NOAA station within {radius_km:g} km of {site['site_name']} has "
                           f"{min_coverage:.0%} PRCP/TMAX/TMIN coverage. Best: {best[0]:.1%}" if best else
                           f"No NOAA station found within {radius_km:g} km of {site['site_name']}.")
    coverage, neg_km, station, weather = best
    mapping = {"site_number": site["site_number"], "station_id": station["station_id"],
               "station_name": station["station_name"], "distance_km": round(-neg_km, 2),
               "complete_weather_coverage": round(coverage, 4)}
    rows = [{"site_number": site["site_number"], "station_id": station["station_id"], **row}
            for row in weather]
    return mapping, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--radius-km", type=float, default=100)
    parser.add_argument("--candidate-stations", type=int, default=20)
    parser.add_argument("--min-weather-coverage", type=float, default=0.80)
    parser.add_argument("--refresh", action="store_true", help="Refetch cached NOAA station files")
    args = parser.parse_args()
    try:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        if start > end or args.candidate_stations < 1 or not 0 < args.min_weather_coverage <= 1:
            parser.error("Check dates, candidate count, and weather coverage (0 to 1).")
        raw_dir = args.output_dir / "raw"
        stations = ghcn_stations(cached(raw_dir / "noaa" / "ghcnd-stations.txt",
                                        f"{GHCN}/ghcnd-stations.txt", args.refresh))
        inventory = cached(raw_dir / "noaa" / "ghcnd-inventory.txt",
                           f"{GHCN}/ghcnd-inventory.txt", args.refresh)
        eligible = eligible_station_ids(inventory, start, end, args.min_weather_coverage)
        sites, flow, mappings, weather = [], [], [], []
        for name, number in SITES.items():
            site = get_site(name, number)
            site_flow = get_flow(site, start, end)
            mapping, site_weather = choose_weather(site, stations, eligible, raw_dir, start, end,
                                                   args.radius_km, args.candidate_stations,
                                                   args.min_weather_coverage, args.refresh)
            sites.append(site); flow.extend(site_flow); mappings.append(mapping); weather.extend(site_weather)
            print(f"{name}: {len(site_flow)} flow days; NOAA {mapping['station_id']} "
                  f"({mapping['distance_km']} km, {mapping['complete_weather_coverage']:.0%} coverage)", file=sys.stderr)
        write_csv(raw_dir / "usgs_sites.csv", list(sites[0]), sites)
        write_csv(raw_dir / "usgs_daily.csv", ["site_number", "date", "flow_cfs", "approval_status", "qualifier"], flow)
        write_csv(raw_dir / "station_mapping.csv", list(mappings[0]), mappings)
        write_csv(raw_dir / "noaa_daily.csv", ["site_number", "station_id", "date", "prcp_mm", "tmax_c", "tmin_c", "snow_mm", "snwd_mm", "pressure_hpa"], weather)
        print(f"Wrote raw data to {raw_dir}")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
