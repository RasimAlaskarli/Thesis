#!/usr/bin/env python3
"""Chapter 4 analysis: computes the per-country dataset, regional
aggregates, timeline phases, and generates the four scatter plots used
in the Results and Discussion chapter.

Inputs:
  - chartData_annual.json   (produced by build_chart_data.py)

Outputs:
  - country_dataset.csv     (one row per country, all derived metrics)
  - figures/fig_4_1_timeline.pdf
  - figures/fig_4_2_urbanization.pdf
  - figures/fig_4_3_median_age.pdf
  - figures/fig_4_4_unemployment.pdf

Also prints to stdout all the headline numbers needed for the prose.

Notes on the unemployment window:
  Unlike urbanization and median age which use the full 1960-2010 window,
  unemployment uses a fixed 1991-2010 window. World Bank does not publish
  unemployment values for most of Europe before 1991, and the 1991
  baseline aligns with the post-Cold-War period that drives the migration
  story in this thesis. Liechtenstein is the only country excluded
  because it has no unemployment data at all in the WDI dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------
# Country definitions
# ----------------------------------------------------------------------------

WESTERN = {
    "AUT": "Austria", "BEL": "Belgium", "CYP": "Cyprus", "DNK": "Denmark",
    "FIN": "Finland", "FRA": "France", "DEU": "Germany", "GRC": "Greece",
    "ISL": "Iceland", "IRL": "Ireland", "ITA": "Italy", "LIE": "Liechtenstein",
    "LUX": "Luxembourg", "MLT": "Malta", "NLD": "Netherlands", "NOR": "Norway",
    "PRT": "Portugal", "ESP": "Spain", "SWE": "Sweden", "CHE": "Switzerland",
    "GBR": "United Kingdom",
}

EASTERN = {
    "ALB": "Albania", "BIH": "Bosnia and Herzegovina", "BGR": "Bulgaria",
    "HRV": "Croatia", "CZE": "Czech Republic", "EST": "Estonia",
    "HUN": "Hungary", "LVA": "Latvia", "LTU": "Lithuania", "MDA": "Moldova",
    "MNE": "Montenegro", "MKD": "North Macedonia", "POL": "Poland",
    "ROU": "Romania", "SRB": "Serbia", "SVK": "Slovakia", "SVN": "Slovenia",
}

START_YEAR = 1960
END_YEAR = 2010

# Unemployment uses a different window: data is not reliably available
# before 1991 for most countries, and the 1991 baseline aligns with the
# post-Cold-War migration period.
UNEMPLOYMENT_START_YEAR = 1991
UNEMPLOYMENT_END_YEAR = 2010


# ----------------------------------------------------------------------------
# Style — match the application's color palette
# ----------------------------------------------------------------------------

COLOR_WEST = "#4878a8"   # blue from the application's net-migration scale
COLOR_EAST = "#c44e52"   # red from the same scale
COLOR_TEXT = "#3d3a35"
COLOR_GRID = "#e8e4dc"
COLOR_AXIS = "#8a857a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": COLOR_AXIS,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "xtick.color": COLOR_AXIS,
    "ytick.color": COLOR_AXIS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.5,
})


# ----------------------------------------------------------------------------
# Data loading and per-country derived metrics
# ----------------------------------------------------------------------------

def load_chart_data(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cumulative_net_migration(country: Dict[str, Dict[str, float]],
                             start: int = START_YEAR,
                             end: int = END_YEAR) -> Optional[float]:
    """Sum annual net migration values from start to end, inclusive."""
    nm = country.get("netMigration", {})
    total = 0.0
    found_any = False
    for y in range(start, end + 1):
        v = nm.get(str(y))
        if v is not None:
            total += v
            found_any = True
    return total if found_any else None


def indicator_change(country: Dict[str, Dict[str, float]],
                     indicator: str,
                     start: int = START_YEAR,
                     end: int = END_YEAR,
                     window: int = 5) -> Optional[float]:
    """Compute end-value minus start-value for an indicator.

    Tolerates missing exact start/end years by looking within `window` years
    on either side. Returns None if no usable start or end value is found.
    """
    series = country.get(indicator, {})
    if not series:
        return None

    start_val = None
    for offset in range(window + 1):
        for y in (start + offset, start - offset):
            if str(y) in series:
                start_val = series[str(y)]
                break
        if start_val is not None:
            break

    end_val = None
    for offset in range(window + 1):
        for y in (end - offset, end + offset):
            if str(y) in series:
                end_val = series[str(y)]
                break
        if end_val is not None:
            break

    if start_val is None or end_val is None:
        return None
    return end_val - start_val


def unemployment_change_fixed(country: Dict[str, Dict[str, float]],
                              start: int = UNEMPLOYMENT_START_YEAR,
                              end: int = UNEMPLOYMENT_END_YEAR
                              ) -> Optional[Tuple[float, float, float]]:
    """Compute end-value minus start-value for unemployment using a fixed
    window. Unlike `indicator_change`, this does not look at neighboring
    years: a country must have data for both `start` and `end` exactly,
    or it is excluded.

    Returns (change, start_val, end_val) or None if either endpoint is
    missing.
    """
    series = country.get("unemployment", {})
    if not series:
        return None
    start_val = series.get(str(start))
    end_val = series.get(str(end))
    if start_val is None or end_val is None:
        return None
    return end_val - start_val, start_val, end_val


def build_country_dataset(chart_data: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict]:
    """One row per country in scope, with all derived metrics."""
    rows = []
    for region_name, codes in [("Western", WESTERN), ("Eastern", EASTERN)]:
        for code, name in codes.items():
            country = chart_data.get(code, {})
            unempl_result = unemployment_change_fixed(country)
            row = {
                "code": code,
                "name": name,
                "region": region_name,
                "cumulative_net_migration": cumulative_net_migration(country),
                "urbanization_change": indicator_change(country, "urbanization"),
                "median_age_change": indicator_change(country, "medianAge"),
                "unemployment_change": unempl_result[0] if unempl_result else None,
                "unemployment_1991": unempl_result[1] if unempl_result else None,
                "unemployment_2010": unempl_result[2] if unempl_result else None,
                "urbanization_1960": country.get("urbanization", {}).get(str(START_YEAR)),
                "urbanization_2010": country.get("urbanization", {}).get(str(END_YEAR)),
                "median_age_1960": country.get("medianAge", {}).get(str(START_YEAR)),
                "median_age_2010": country.get("medianAge", {}).get(str(END_YEAR)),
                "population_1960": country.get("population", {}).get(str(START_YEAR)),
                "population_2010": country.get("population", {}).get(str(END_YEAR)),
            }
            rows.append(row)
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ----------------------------------------------------------------------------
# Aggregates and headline numbers
# ----------------------------------------------------------------------------

def regional_total(rows: List[Dict], region: str, key: str) -> float:
    return sum(r[key] for r in rows if r["region"] == region and r[key] is not None)


def regional_mean(rows: List[Dict], region: str, key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r["region"] == region and r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def timeline_phase_total(chart_data: Dict, codes: Dict[str, str],
                         start: int, end: int) -> float:
    """Sum of net migration across all listed countries from start to end."""
    total = 0.0
    for code in codes:
        nm = chart_data.get(code, {}).get("netMigration", {})
        for y in range(start, end + 1):
            v = nm.get(str(y))
            if v is not None:
                total += v
    return total


def correlation(rows: List[Dict], x_key: str, y_key: str) -> Optional[Tuple[float, int]]:
    """Pearson correlation between two metrics, plus N (countries with both values)."""
    pairs = [(r[x_key], r[y_key]) for r in rows
             if r[x_key] is not None and r[y_key] is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    return float(np.corrcoef(xs, ys)[0, 1]), len(pairs)


def top_n_by(rows: List[Dict], key: str, n: int = 5,
             reverse: bool = True, region: Optional[str] = None) -> List[Dict]:
    candidates = [r for r in rows if r[key] is not None]
    if region:
        candidates = [r for r in candidates if r["region"] == region]
    return sorted(candidates, key=lambda r: r[key], reverse=reverse)[:n]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def fig_timeline(chart_data: Dict, output_path: Path) -> None:
    """Cumulative annual net migration by region over 1960-2010."""
    years = list(range(START_YEAR, END_YEAR + 1))

    def annual_regional_total(codes: Dict[str, str], year: int) -> float:
        return sum(chart_data.get(c, {}).get("netMigration", {}).get(str(year), 0)
                   for c in codes)

    west_annual = [annual_regional_total(WESTERN, y) for y in years]
    east_annual = [annual_regional_total(EASTERN, y) for y in years]

    # Cumulative
    west_cum = np.cumsum(west_annual)
    east_cum = np.cumsum(east_annual)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, np.array(west_cum) / 1e6, color=COLOR_WEST, linewidth=2,
            label=f"Western Europe ({len(WESTERN)} countries)")
    ax.plot(years, np.array(east_cum) / 1e6, color=COLOR_EAST, linewidth=2,
            label=f"Eastern Europe ({len(EASTERN)} countries)")
    ax.axhline(0, color=COLOR_AXIS, linewidth=0.5, linestyle="-", alpha=0.5)

    # Mark key dates
    for year, label in [(1989, "Fall of\ncommunism"), (2004, "EU\nenlargement")]:
        ax.axvline(year, color=COLOR_AXIS, linewidth=0.5, linestyle="--", alpha=0.5)
        ax.text(year, ax.get_ylim()[1] if False else 28, label,
                fontsize=7, color=COLOR_AXIS, ha="center", va="top")

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative net migration since 1960 (millions)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(START_YEAR, END_YEAR)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def fig_scatter(rows: List[Dict], x_key: str, y_key: str,
                x_label: str, y_label: str, output_path: Path,
                x_scale: float = 1.0) -> None:
    """Scatter plot with country labels and per-region regression lines."""
    pairs = [(r[x_key] / x_scale, r[y_key], r["region"], r["name"], r["code"])
             for r in rows
             if r[x_key] is not None and r[y_key] is not None]

    if len(pairs) < 3:
        print(f"  WARNING: only {len(pairs)} points for {x_key} vs {y_key}, skipping figure")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot points
    for x, y, region, name, code in pairs:
        color = COLOR_WEST if region == "Western" else COLOR_EAST
        ax.scatter(x, y, s=35, color=color, alpha=0.75, edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.annotate(code, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=6, color=COLOR_TEXT, alpha=0.85)

    # Helper: fit and plot a regression line for one group of points
    def fit_and_plot(group_pairs, color, linestyle, alpha, zorder):
        if len(group_pairs) < 3:
            return None
        xs = np.array([p[0] for p in group_pairs])
        ys = np.array([p[1] for p in group_pairs])
        if np.std(xs) == 0:
            return None
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_line, m * x_line + b, color=color, linewidth=1.2,
                linestyle=linestyle, alpha=alpha, zorder=zorder)
        r = np.corrcoef(xs, ys)[0, 1]
        return r, len(group_pairs)

    # Compute overall r for the title (no line drawn)
    xs_all = np.array([p[0] for p in pairs])
    ys_all = np.array([p[1] for p in pairs])
    overall_r = np.corrcoef(xs_all, ys_all)[0, 1] if np.std(xs_all) > 0 else None
    overall_n = len(pairs)

    # Per-region regressions
    western_pairs = [p for p in pairs if p[2] == "Western"]
    eastern_pairs = [p for p in pairs if p[2] == "Eastern"]
    west_stats = fit_and_plot(western_pairs, COLOR_WEST, "-", 0.85, 2)
    east_stats = fit_and_plot(eastern_pairs, COLOR_EAST, "-", 0.85, 2)

    # Title summarising all three correlations
    title_parts = []
    if overall_r is not None:
        title_parts.append(f"Overall r = {overall_r:.2f}  (n = {overall_n})")
    if west_stats is not None:
        title_parts.append(f"WE r = {west_stats[0]:.2f}  (n = {west_stats[1]})")
    if east_stats is not None:
        title_parts.append(f"EE r = {east_stats[0]:.2f}  (n = {east_stats[1]})")
    if title_parts:
        ax.set_title("   ".join(title_parts),
                     fontsize=9, color=COLOR_TEXT, loc="left", pad=8)

    ax.axvline(0, color=COLOR_AXIS, linewidth=0.5, alpha=0.4, zorder=1)
    ax.axhline(0, color=COLOR_AXIS, linewidth=0.5, alpha=0.4, zorder=1)

    # Y-axis padding (initial)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.03 * y_range, y_max + 0.07 * y_range)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_WEST,
               markersize=7, label="Western Europe"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_EAST,
               markersize=7, label="Eastern Europe"),
        Line2D([0], [0], color=COLOR_WEST, linewidth=1.2,
               label="WE trend"),
        Line2D([0], [0], color=COLOR_EAST, linewidth=1.2,
               label="EE trend"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=7)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.4)

    # Final y-axis padding
    y_vals = [p[1] for p in pairs]
    y_min, y_max = min(y_vals), max(y_vals)
    y_margin = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def fmt_millions(n: float) -> str:
    return f"{n/1e6:+.1f} million" if n != 0 else "0"


def fmt_thousands(n: float) -> str:
    return f"{n/1e3:+.0f}K" if abs(n) >= 1e3 else f"{n:+.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="chartData_annual.json")
    parser.add_argument("--out-dataset", default="country_dataset.csv")
    parser.add_argument("--out-figures", default="figures")
    args = parser.parse_args()

    chart_data = load_chart_data(Path(args.input))
    rows = build_country_dataset(chart_data)
    write_csv(rows, Path(args.out_dataset))

    figures_dir = Path(args.out_figures)
    figures_dir.mkdir(exist_ok=True)

    # Figures
    fig_timeline(chart_data, figures_dir / "fig_4_1_timeline.pdf")

    fig_scatter(rows, "cumulative_net_migration", "urbanization_change",
                "Cumulative net migration, 1960–2010 (millions)",
                "Change in urbanization rate (percentage points)",
                figures_dir / "fig_4_2_urbanization.pdf",
                x_scale=1e6)

    fig_scatter(rows, "cumulative_net_migration", "median_age_change",
                "Cumulative net migration, 1960–2010 (millions)",
                "Change in median age (years)",
                figures_dir / "fig_4_3_median_age.pdf",
                x_scale=1e6)

    fig_scatter(rows, "cumulative_net_migration", "unemployment_change",
                "Cumulative net migration, 1960–2010 (millions)",
                f"Change in unemployment rate, "
                f"{UNEMPLOYMENT_START_YEAR}\u2013{UNEMPLOYMENT_END_YEAR} (pp)",
                figures_dir / "fig_4_4_unemployment.pdf",
                x_scale=1e6)

    # ===== Headline numbers for the prose =====
    print("\n" + "=" * 70)
    print("HEADLINE NUMBERS FOR CHAPTER 4")
    print("=" * 70)

    print("\n--- 4.1 Overview ---\n")
    west_total = regional_total(rows, "Western", "cumulative_net_migration")
    east_total = regional_total(rows, "Eastern", "cumulative_net_migration")
    print(f"Western Europe total net migration 1960-2010: {fmt_millions(west_total)}")
    print(f"Eastern Europe total net migration 1960-2010: {fmt_millions(east_total)}")
    print(f"Gap: {fmt_millions(west_total - east_total)}")

    print("\n  Timeline phases (Eastern Europe):")
    east_pre89 = timeline_phase_total(chart_data, EASTERN, 1960, 1989)
    east_post89 = timeline_phase_total(chart_data, EASTERN, 1990, 2003)
    east_post04 = timeline_phase_total(chart_data, EASTERN, 2004, 2010)
    print(f"    1960-1989: {fmt_millions(east_pre89)}")
    print(f"    1990-2003: {fmt_millions(east_post89)}")
    print(f"    2004-2010: {fmt_millions(east_post04)}")

    print("\n  Timeline phases (Western Europe):")
    west_pre89 = timeline_phase_total(chart_data, WESTERN, 1960, 1989)
    west_post89 = timeline_phase_total(chart_data, WESTERN, 1990, 2003)
    west_post04 = timeline_phase_total(chart_data, WESTERN, 2004, 2010)
    print(f"    1960-1989: {fmt_millions(west_pre89)}")
    print(f"    1990-2003: {fmt_millions(west_post89)}")
    print(f"    2004-2010: {fmt_millions(west_post04)}")

    print("\n  Top 5 receiving countries (by cumulative net migration):")
    for r in top_n_by(rows, "cumulative_net_migration", n=5, reverse=True):
        print(f"    {r['name']:<25} {fmt_millions(r['cumulative_net_migration'])}")

    print("\n  Top 5 sending countries (by most negative cumulative net migration):")
    for r in top_n_by(rows, "cumulative_net_migration", n=5, reverse=False):
        print(f"    {r['name']:<25} {fmt_millions(r['cumulative_net_migration'])}")

    print("\n--- 4.2 Urbanization ---\n")
    print(f"Western avg urbanization in 1960: {regional_mean(rows, 'Western', 'urbanization_1960'):.1f}%")
    print(f"Western avg urbanization in 2010: {regional_mean(rows, 'Western', 'urbanization_2010'):.1f}%")
    print(f"Western avg change:              {regional_mean(rows, 'Western', 'urbanization_change'):+.1f} pp")
    print(f"Eastern avg urbanization in 1960: {regional_mean(rows, 'Eastern', 'urbanization_1960'):.1f}%")
    print(f"Eastern avg urbanization in 2010: {regional_mean(rows, 'Eastern', 'urbanization_2010'):.1f}%")
    print(f"Eastern avg change:              {regional_mean(rows, 'Eastern', 'urbanization_change'):+.1f} pp")
    cor = correlation(rows, "cumulative_net_migration", "urbanization_change")
    if cor:
        print(f"\nCorrelation (net migration vs urbanization change): r = {cor[0]:.3f}, n = {cor[1]}")

    print("\n--- 4.3 Median Age ---\n")
    print(f"Western avg median age in 1960: {regional_mean(rows, 'Western', 'median_age_1960'):.1f} yrs")
    print(f"Western avg median age in 2010: {regional_mean(rows, 'Western', 'median_age_2010'):.1f} yrs")
    print(f"Western avg change:             {regional_mean(rows, 'Western', 'median_age_change'):+.1f} yrs")
    print(f"Eastern avg median age in 1960: {regional_mean(rows, 'Eastern', 'median_age_1960'):.1f} yrs")
    print(f"Eastern avg median age in 2010: {regional_mean(rows, 'Eastern', 'median_age_2010'):.1f} yrs")
    print(f"Eastern avg change:             {regional_mean(rows, 'Eastern', 'median_age_change'):+.1f} yrs")
    cor = correlation(rows, "cumulative_net_migration", "median_age_change")
    if cor:
        print(f"\nCorrelation (net migration vs median age change): r = {cor[0]:.3f}, n = {cor[1]}")

    print(f"\n--- 4.4 Unemployment ({UNEMPLOYMENT_START_YEAR}\u2013{UNEMPLOYMENT_END_YEAR}) ---\n")
    n_with_unempl = sum(1 for r in rows if r["unemployment_change"] is not None)
    print(f"Countries with unemployment change data: {n_with_unempl} of {len(rows)}")
    print(f"  (fixed window: requires data for both {UNEMPLOYMENT_START_YEAR} and {UNEMPLOYMENT_END_YEAR})")

    excluded = [r for r in rows if r["unemployment_change"] is None]
    if excluded:
        print(f"\n  Countries excluded (missing {UNEMPLOYMENT_START_YEAR} or {UNEMPLOYMENT_END_YEAR}):")
        for r in excluded:
            row = chart_data.get(r["code"], {}).get("unemployment", {})
            v91 = row.get(str(UNEMPLOYMENT_START_YEAR))
            v10 = row.get(str(UNEMPLOYMENT_END_YEAR))
            v91_str = f"{v91:.1f}" if v91 is not None else "missing"
            v10_str = f"{v10:.1f}" if v10 is not None else "missing"
            print(f"    {r['name']:<25} 1991={v91_str}  2010={v10_str}")

    print(f"\n  Western avg unemployment in {UNEMPLOYMENT_START_YEAR}: "
          f"{regional_mean(rows, 'Western', 'unemployment_1991'):.1f}%")
    print(f"  Western avg unemployment in {UNEMPLOYMENT_END_YEAR}: "
          f"{regional_mean(rows, 'Western', 'unemployment_2010'):.1f}%")
    print(f"  Western avg change:             "
          f"{regional_mean(rows, 'Western', 'unemployment_change'):+.1f} pp")
    print(f"  Eastern avg unemployment in {UNEMPLOYMENT_START_YEAR}: "
          f"{regional_mean(rows, 'Eastern', 'unemployment_1991'):.1f}%")
    print(f"  Eastern avg unemployment in {UNEMPLOYMENT_END_YEAR}: "
          f"{regional_mean(rows, 'Eastern', 'unemployment_2010'):.1f}%")
    print(f"  Eastern avg change:             "
          f"{regional_mean(rows, 'Eastern', 'unemployment_change'):+.1f} pp")

    cor = correlation(rows, "cumulative_net_migration", "unemployment_change")
    if cor:
        print(f"\n  Correlation (net migration vs unemployment change): "
              f"r = {cor[0]:.3f}, n = {cor[1]}")

    print("\n  Top 5 largest unemployment decreases:")
    for r in top_n_by(rows, "unemployment_change", n=5, reverse=False):
        print(f"    {r['name']:<25} {r['unemployment_change']:+.1f} pp  "
              f"({r['unemployment_1991']:.1f}% -> {r['unemployment_2010']:.1f}%)")
    print("\n  Top 5 largest unemployment increases:")
    for r in top_n_by(rows, "unemployment_change", n=5, reverse=True):
        print(f"    {r['name']:<25} {r['unemployment_change']:+.1f} pp  "
              f"({r['unemployment_1991']:.1f}% -> {r['unemployment_2010']:.1f}%)")

    print("\n" + "=" * 70)
    print(f"\n[OK] wrote {args.out_dataset}")
    print(f"[OK] wrote 4 figures to {figures_dir}/")


if __name__ == "__main__":
    main()