#!/usr/bin/env python3
"""Build per-country demographic and economic indicator data for the
application, producing both an annual-resolution and a 5-year-resolution
output file.

Inputs:
  World Bank WDI CSVs (wide format, downloaded from data.worldbank.org):
    - net_migration.csv  (SM.POP.NETM)
    - urbanization.csv   (SP.URB.TOTL.IN.ZS)
    - unemployment.csv   (SL.UEM.TOTL.NE.ZS)
    - population.csv     (SP.POP.TOTL)

  Long-format CSV (one row per country-year, e.g. from Our World in Data
  or UN WPP exports):
    - median_age.csv     (UN World Population Prospects)

Outputs:
  - chartData_annual.json  — every available year per indicator
  - chartData_5yr.json     — only years on the 5-year grid (1960, 1965, ...)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional


# Indicator label -> (filename, file_format).
#   "wdi"  = World Bank wide format with metadata rows then header row
#   "long" = one row per country-year (Entity, Code, Year, value columns)
INDICATORS_REQUIRED = [
    ("netMigration", "net_migration.csv", "wdi"),
    ("urbanization", "urbanization.csv",  "wdi"),
    ("unemployment", "unemployment.csv",  "wdi"),
    ("population",   "population.csv",    "wdi"),
    ("medianAge",    "median_age.csv",    "long"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=".", help="Directory containing the raw CSV files")
    p.add_argument("--out-annual", default="chartData_annual.json", help="Path to annual output JSON")
    p.add_argument("--out-5yr",    default="chartData_5yr.json",   help="Path to 5-year output JSON")
    p.add_argument("--indent", type=int, default=2)
    p.add_argument("--start-year", type=int, default=1960)
    p.add_argument("--end-year",   type=int, default=2010)
    return p.parse_args()


def safe_number(text: str) -> Optional[float]:
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_wdi_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse a World Bank wide-format CSV into {country_code: {year: value}}."""
    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "Country Name":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find header row in {path}")

    header = rows[header_idx]
    code_idx = header.index("Country Code")
    year_columns = [(c, i) for i, c in enumerate(header) if c.isdigit()]

    for row in rows[header_idx + 1:]:
        if len(row) <= code_idx:
            continue
        code = row[code_idx].strip().upper()
        if not code or len(code) != 3:
            continue

        country_data: Dict[str, float] = {}
        for year_str, idx in year_columns:
            if idx >= len(row):
                continue
            value = safe_number(row[idx])
            if value is None:
                continue
            country_data[year_str] = value

        if country_data:
            out[code] = country_data

    return out


def load_long_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse a long-format CSV (Entity, Code, Year, value...).

    Used for median age data which arrives from UN/OWID with one row per
    country-year. If multiple value columns exist (e.g. 'estimates' and
    'medium' variant projections), tries them in column order and uses the
    first non-blank value for each row.
    """
    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read header row in {path}")

        meta_fields = {"Entity", "Code", "Year"}
        value_fields = [c for c in reader.fieldnames if c not in meta_fields]
        if not value_fields:
            raise ValueError(f"No value columns found in {path}")

        for row in reader:
            code = (row.get("Code") or "").strip().upper()
            year = (row.get("Year") or "").strip()
            if not code or len(code) != 3 or not year.isdigit():
                continue

            value: Optional[float] = None
            for vf in value_fields:
                v = safe_number(row.get(vf))
                if v is not None:
                    value = v
                    break

            if value is None:
                continue

            out.setdefault(code, {})[year] = value

    return out


def round_for_indicator(indicator: str, value: float) -> float | int:
    """Round each indicator to a sensible precision for the output file."""
    if indicator in ("netMigration", "population"):
        return int(round(value))
    return round(value, 1)


def restrict_to_years(country_dict: Dict[str, float], allowed_years: set[str]) -> Dict[str, float]:
    return {y: v for y, v in country_dict.items() if y in allowed_years}


def merge_indicators(
    indicators: Dict[str, Dict[str, Dict[str, float]]],
    year_filter: Optional[set[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float | int]]]:
    """Combine all indicators into the per-country output structure."""
    all_codes: set[str] = set()
    for ind_data in indicators.values():
        all_codes.update(ind_data.keys())

    out: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for code in sorted(all_codes):
        country_block: Dict[str, Dict[str, float | int]] = {}
        for ind_name, ind_data in indicators.items():
            country_data = ind_data.get(code, {})
            if year_filter is not None:
                country_data = restrict_to_years(country_data, year_filter)
            if not country_data:
                continue
            country_block[ind_name] = {
                y: round_for_indicator(ind_name, v) for y, v in country_data.items()
            }
        if country_block:
            out[code] = country_block
    return out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)

    indicators: Dict[str, Dict[str, Dict[str, float]]] = {}

    for label, filename, fmt in INDICATORS_REQUIRED:
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}. "
                f"Place the raw CSV in {input_dir}."
            )

        if fmt == "wdi":
            indicators[label] = load_wdi_csv(path)
        elif fmt == "long":
            indicators[label] = load_long_csv(path)
        else:
            raise ValueError(f"Unknown file format '{fmt}' for {label}")

        print(f"[OK] loaded {label}: {len(indicators[label])} countries from {filename}")

    annual_year_filter = {str(y) for y in range(args.start_year, args.end_year + 1)}
    annual_output = merge_indicators(indicators, annual_year_filter)

    five_year_filter = {
        str(y) for y in range(args.start_year, args.end_year + 1) if y % 5 == 0
    }
    five_year_output = merge_indicators(indicators, five_year_filter)

    Path(args.out_annual).write_text(
        json.dumps(annual_output, ensure_ascii=False, indent=args.indent),
        encoding="utf-8",
    )
    Path(args.out_5yr).write_text(
        json.dumps(five_year_output, ensure_ascii=False, indent=args.indent),
        encoding="utf-8",
    )

    print(f"\n[OK] wrote {args.out_annual}: {len(annual_output)} countries")
    print(f"[OK] wrote {args.out_5yr}:    {len(five_year_output)} countries")


if __name__ == "__main__":
    main()