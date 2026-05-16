"""
Validation script for the application's migration data.

Compares two sources of net migration that come from completely independent
production processes, to check that the bilateral flow data shipped with the
application is consistent with World Bank net migration:

  1. Sum of bilateral flows: for each country c and 5-year period p,
        abel_net(c, p) = ti(c, p) - to(c, p)
     where ti and to are the totals of all incoming/outgoing bilateral flows
     in migrationData_5yr.json. These are derived from Abel's stock-differencing
     and demographic-accounting methods.

  2. World Bank net migration (SM.POP.NETM): annual values from chartData.json,
     summed over the same 5-year window. Produced by UN DESA and reported
     through the World Bank's World Development Indicators.

For each (country, period) pair we report the two values, their ratio,
and aggregate statistics (Pearson correlation, sign agreement, ratio
quartiles). Cumulative country totals over 1960-2009 are also reported.

Usage:
    python validate_migration_data.py

By default the script looks for migrationData_5yr.json and chartData.json
in the working directory. Override with --bilateral and --chart.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Country groupings used in Chapter 4 of the thesis.
WESTERN = [
    "AUT", "BEL", "CYP", "DNK", "FIN", "FRA", "DEU", "GRC", "ISL", "IRL",
    "ITA", "LIE", "LUX", "MLT", "NLD", "NOR", "PRT", "ESP", "SWE", "CHE", "GBR",
]
EASTERN = [
    "ALB", "BIH", "BGR", "HRV", "CZE", "EST", "HUN", "LVA", "LTU", "MDA",
    "MNE", "MKD", "POL", "ROU", "SRB", "SVK", "SVN",
]
ALL_COUNTRIES = WESTERN + EASTERN

# 5-year periods covered by the bilateral flow data.
PERIODS = [str(y) for y in range(1960, 2010, 5)]


def pearson(xs, ys):
    """Pearson correlation coefficient. Returns None if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def quantile(sorted_xs, q):
    """Linear-interpolated quantile on an already-sorted list."""
    if not sorted_xs:
        return None
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def abel_net(bilateral, country, period):
    """Bilateral-derived net migration: total inflows minus total outflows."""
    data = bilateral.get(period, {}).get(country, {})
    return data.get("ti", 0) - data.get("to", 0)


def wb_net_5yr(chart, country, period_start):
    """World Bank net migration summed over the 5-year window."""
    start = int(period_start)
    series = chart.get(country, {}).get("netMigration", {})
    return sum(series.get(str(y), 0) or 0 for y in range(start, start + 5))


def collect_observations(bilateral, chart):
    """Return list of (country, period, abel_net, wb_net) tuples."""
    rows = []
    for cc in ALL_COUNTRIES:
        for p in PERIODS:
            ab = abel_net(bilateral, cc, p)
            wb = wb_net_5yr(chart, cc, p)
            if ab != 0 or wb != 0:
                rows.append((cc, p, ab, wb))
    return rows


def period_level_summary(rows):
    """Statistics across every country-period observation."""
    ab = [r[2] for r in rows]
    wb = [r[3] for r in rows]
    r = pearson(ab, wb)

    nonzero = [(a, w) for a, w in zip(ab, wb) if a != 0 and w != 0]
    sign_agree = sum(1 for a, w in nonzero if (a > 0) == (w > 0))

    # Ratios only meaningful when both sides are non-trivial.
    ratios = sorted(a / w for a, w in zip(ab, wb)
                    if abs(w) > 10_000 and abs(a) > 10_000)

    return {
        "n_obs": len(rows),
        "pearson_r": r,
        "sign_agree": sign_agree,
        "sign_total": len(nonzero),
        "ratio_n": len(ratios),
        "ratio_median": quantile(ratios, 0.5) if ratios else None,
        "ratio_q1": quantile(ratios, 0.25) if ratios else None,
        "ratio_q3": quantile(ratios, 0.75) if ratios else None,
    }


def country_level_summary(bilateral, chart):
    """Cumulative 1960-2009 totals per country."""
    rows = []
    for cc in ALL_COUNTRIES:
        region = "West" if cc in WESTERN else "East"
        ab_cum = sum(abel_net(bilateral, cc, p) for p in PERIODS)
        wb_cum = sum(wb_net_5yr(chart, cc, p) for p in PERIODS)
        ratio = ab_cum / wb_cum if abs(wb_cum) > 100 else None
        rows.append({
            "country": cc,
            "region": region,
            "abel_cum": ab_cum,
            "wb_cum": wb_cum,
            "diff": ab_cum - wb_cum,
            "ratio": ratio,
        })
    return rows


def print_period_table(rows, limit=None):
    """Pretty-print the per-country-period comparison."""
    print(f"{'Country':<8} {'Period':<10} {'Abel net':>14} {'WB net':>14} "
          f"{'Diff':>14} {'Ratio':>8}")
    print("-" * 80)
    shown = 0
    for cc, p, ab, wb in rows:
        if limit is not None and shown >= limit:
            print(f"... ({len(rows) - shown} more rows hidden)")
            break
        ratio_str = f"{ab/wb:+.2f}" if abs(wb) > 100 else "    n/a"
        period_label = f"{p}-{int(p)+4}"
        print(f"{cc:<8} {period_label:<10} {ab:>+14,} {wb:>+14,} "
              f"{ab-wb:>+14,} {ratio_str:>8}")
        shown += 1


def print_country_table(country_rows):
    """Pretty-print the cumulative-by-country comparison."""
    print(f"{'Country':<8} {'Region':<8} {'Abel cum':>15} {'WB cum':>15} "
          f"{'Diff':>15} {'Ratio':>8}")
    print("-" * 80)
    # Sort by absolute WB total so the most important countries appear first.
    for row in sorted(country_rows, key=lambda r: -abs(r["wb_cum"])):
        ratio_str = f"{row['ratio']:+.2f}" if row["ratio"] is not None else "    n/a"
        print(f"{row['country']:<8} {row['region']:<8} "
              f"{row['abel_cum']:>+15,} {row['wb_cum']:>+15,} "
              f"{row['diff']:>+15,} {ratio_str:>8}")

    # Region totals.
    w_ab = sum(r["abel_cum"] for r in country_rows if r["region"] == "West")
    w_wb = sum(r["wb_cum"] for r in country_rows if r["region"] == "West")
    e_ab = sum(r["abel_cum"] for r in country_rows if r["region"] == "East")
    e_wb = sum(r["wb_cum"] for r in country_rows if r["region"] == "East")
    print("-" * 80)
    print(f"{'WEST':<8} {'(21)':<8} {w_ab:>+15,} {w_wb:>+15,} "
          f"{w_ab-w_wb:>+15,}")
    print(f"{'EAST':<8} {'(17)':<8} {e_ab:>+15,} {e_wb:>+15,} "
          f"{e_ab-e_wb:>+15,}")


def print_summary(period_summary, country_rows):
    print()
    print("=" * 80)
    print("PERIOD-LEVEL SUMMARY (all 38 countries × 10 five-year periods)")
    print("=" * 80)
    s = period_summary
    print(f"  Observations:       {s['n_obs']}")
    if s["pearson_r"] is not None:
        print(f"  Pearson r:          {s['pearson_r']:.3f}")
    print(f"  Sign agreement:     {s['sign_agree']}/{s['sign_total']} "
          f"({100*s['sign_agree']/s['sign_total']:.1f}%)")
    print(f"  Ratio Abel/WB (only when both |val| > 10,000):")
    print(f"    n     = {s['ratio_n']}")
    print(f"    Q1    = {s['ratio_q1']:+.2f}")
    print(f"    Med   = {s['ratio_median']:+.2f}")
    print(f"    Q3    = {s['ratio_q3']:+.2f}")

    print()
    print("=" * 80)
    print("COUNTRY-LEVEL SUMMARY (cumulative 1960-2009)")
    print("=" * 80)
    ab_cum = [r["abel_cum"] for r in country_rows]
    wb_cum = [r["wb_cum"] for r in country_rows]
    r = pearson(ab_cum, wb_cum)
    nonzero = [(a, w) for a, w in zip(ab_cum, wb_cum) if a != 0 and w != 0]
    sign_agree = sum(1 for a, w in nonzero if (a > 0) == (w > 0))
    print(f"  Countries:          {len(country_rows)}")
    if r is not None:
        print(f"  Pearson r:          {r:.3f}")
    print(f"  Sign agreement:     {sign_agree}/{len(nonzero)}")

    # Total ratio.
    w_ab = sum(r["abel_cum"] for r in country_rows if r["region"] == "West")
    w_wb = sum(r["wb_cum"] for r in country_rows if r["region"] == "West")
    e_ab = sum(r["abel_cum"] for r in country_rows if r["region"] == "East")
    e_wb = sum(r["wb_cum"] for r in country_rows if r["region"] == "East")
    print(f"  Western totals:     Abel {w_ab:+,} vs WB {w_wb:+,} "
          f"(ratio {w_ab/w_wb:+.2f})")
    print(f"  Eastern totals:     Abel {e_ab:+,} vs WB {e_wb:+,} "
          f"(ratio {e_ab/e_wb:+.2f})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare bilateral-flow-derived net migration "
                    "against World Bank net migration."
    )
    parser.add_argument(
        "--bilateral",
        type=Path,
        default=Path("migrationData_5yr.json"),
        help="Path to the bilateral migration JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=Path("chartData.json"),
        help="Path to the demographic JSON with World Bank net migration "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--show",
        choices=["summary", "country", "period", "all"],
        default="summary",
        help="What to print. 'summary' (default) prints just the headline stats. "
             "'country' adds the cumulative-by-country table. 'period' adds the "
             "full per-country-period table. 'all' prints everything.",
    )
    parser.add_argument(
        "--period-limit",
        type=int,
        default=50,
        help="Maximum rows shown in the period-level table (default: %(default)s).",
    )
    args = parser.parse_args()

    if not args.bilateral.exists():
        print(f"error: cannot find {args.bilateral}", file=sys.stderr)
        sys.exit(1)
    if not args.chart.exists():
        print(f"error: cannot find {args.chart}", file=sys.stderr)
        sys.exit(1)

    with args.bilateral.open() as f:
        bilateral = json.load(f)
    with args.chart.open() as f:
        chart = json.load(f)

    rows = collect_observations(bilateral, chart)
    if not rows:
        print("error: no observations found - check the input files", file=sys.stderr)
        sys.exit(1)

    period_summary = period_level_summary(rows)
    country_rows = country_level_summary(bilateral, chart)

    if args.show in ("period", "all"):
        print("PER-COUNTRY-PERIOD COMPARISON")
        print("=" * 80)
        print_period_table(rows, limit=args.period_limit)
        print()

    if args.show in ("country", "all"):
        print("CUMULATIVE PER-COUNTRY COMPARISON (1960-2009)")
        print("=" * 80)
        print_country_table(country_rows)

    print_summary(period_summary, country_rows)


if __name__ == "__main__":
    main()
