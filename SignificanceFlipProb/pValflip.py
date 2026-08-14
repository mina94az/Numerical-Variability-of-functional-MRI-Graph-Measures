#!/usr/bin/env python3
"""Generate reproducible p-value significance-flip lookup tables.

The p-value uncertainty is not a single global value: it is read from the
population-specific expanded lookup CSV for each threshold, metric, sample
size, statistic, and statistical test. The script computes nominal p-values
and the appropriate Beta tail probabilities, then writes CSV result tables.

Generate all three population CSV outputs with::

    python SignificanceFlipProb/pValflip.py --all-populations

The batch command writes CSV tables under
``tables/{combined,PD,HC}``. Its authoritative inputs are the corresponding
``table1_all_formulae_lookup_values.csv`` files in those same directories.
This script writes CSV files only; it does not generate LaTeX or PDF files.
The calculation is deterministic: no random values are sampled. The Beta
distribution is fitted analytically by matching the nominal p-value and
propagated p-value variance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, f, t


ALPHA = 0.05
P_VALUE_FORMULAS = {
    "two_sample_t_p_value",
    "partial_correlation_p_value",
    "ancova_p_value",
}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TABLES_DIR = PROJECT_DIR / "tables"
POPULATION_LOOKUP_PATHS = {
    population: TABLES_DIR / population / "table1_all_formulae_lookup_values.csv"
    for population in ("combined", "PD", "HC")
}
DEFAULT_LOOKUP_PATH = POPULATION_LOOKUP_PATHS["combined"]
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "figure4_computed_pvalues.png"
DEFAULT_RESULTS_PATH = TABLES_DIR / "combined" / "p_value_flip_results.csv"
DEFAULT_FLIP_TABLE_CSV = TABLES_DIR / "combined" / "significance_flip_probability.csv"


class ReproducibleHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve command examples while also displaying argument defaults."""


def beta_params_from_mean_var(mu, var, eps=1e-12):
    """Return Beta parameters fitted to a mean and variance.

    A Beta distribution requires ``variance < mean * (1 - mean)``. Values at
    or beyond that boundary are clipped to the nearest valid value.
    """
    mu = np.clip(mu, eps, 1 - eps)
    max_var = np.maximum(mu * (1 - mu) - eps, eps)
    var_clamped = np.minimum(var, max_var)
    denominator = np.where(var_clamped == 0, eps, var_clamped)
    k_calculated = mu * (1 - mu) / denominator - 1.0
    k_calculated = np.maximum(k_calculated, eps)

    a_calculated = mu * k_calculated
    b_calculated = (1 - mu) * k_calculated

    return a_calculated, b_calculated


def flip_proba_beta(p0, sigma_P, alpha=ALPHA):
    """Return the probability that numerical variability crosses ``alpha``."""
    p0 = np.asarray(p0, dtype=float)
    sigma_P = np.asarray(sigma_P, dtype=float)
    zero_variance = sigma_P <= 0
    var_P = sigma_P**2
    a, b = beta_params_from_mean_var(p0, var_P)

    # A significant observed p-value flips when its perturbed value exceeds
    # alpha; a non-significant value flips when the perturbed value is at most
    # alpha.  This is evaluated per lookup row using its own sigma_p.
    cdf_at_alpha = beta.cdf(alpha, a, b)
    pflip = np.where(p0 <= alpha, 1.0 - cdf_at_alpha, cdf_at_alpha)
    pflip = np.where(zero_variance, 0.0, pflip)

    return np.clip(pflip, 0.0, 1.0)


def pvalue_ttest(t_stat, df):
    """Return the two-sided p-value for an already-computed t statistic."""
    return 2 * t.sf(np.abs(t_stat), df)


def pvalue_partial_corr(r, df):
    """Return a two-sided partial-correlation p-value from r and its t df."""
    r = np.asarray(r, dtype=float)
    if np.any(np.abs(r) >= 1):
        raise ValueError("Partial-correlation coefficients must be strictly between -1 and 1.")
    t_stat = r * np.sqrt(df / (1 - r**2))
    return 2 * t.sf(np.abs(t_stat), df)


def pvalue_ancova(F_stat, df2, df1=1):
    """Return the upper-tail p-value for an ANCOVA (F-test) statistic."""
    return f.sf(F_stat, df1, df2)


def load_sigma_p_lookup(lookup_path=DEFAULT_LOOKUP_PATH):
    """Load p-value rows from an expanded uncertainty lookup CSV."""
    lookup_path = Path(lookup_path)
    if not lookup_path.is_file():
        raise FileNotFoundError(
            f"Expanded uncertainty lookup not found: {lookup_path}\n"
            "Generate table1_all_formulae_lookup_values.csv first or pass "
            "--lookup with its path."
        )
    lookup = pd.read_csv(lookup_path)
    required = {"Threshold", "Metric", "NPVR", "n", "uncertainty", "formula"}
    missing = required.difference(lookup.columns)
    if missing:
        raise ValueError(f"Lookup table is missing columns: {sorted(missing)}")

    sigma_p_rows = lookup.loc[lookup["formula"].isin(P_VALUE_FORMULAS)].copy()
    if sigma_p_rows.empty:
        raise ValueError("The lookup table contains no *_p_value formula rows.")
    sigma_p_rows = sigma_p_rows.rename(columns={"uncertainty": "sigma_p"})
    sigma_p_rows["sigma_p"] = pd.to_numeric(sigma_p_rows["sigma_p"], errors="raise")
    return sigma_p_rows


def compute_lookup_pvalues(results):
    """Compute each lookup row's p-value from its formula and parameters."""
    p_values = pd.Series(index=results.index, dtype=float)
    ttest = results["formula"] == "two_sample_t_p_value"
    partial = results["formula"] == "partial_correlation_p_value"
    ancova = results["formula"] == "ancova_p_value"
    statistic = pd.to_numeric(results["statistic"], errors="raise")
    t_df = pd.to_numeric(results.loc[ttest | partial, "df"], errors="raise")
    f_df2 = pd.to_numeric(results.loc[ancova, "df2"], errors="raise")

    if not np.allclose(t_df, results.loc[ttest | partial, "n"]):
        raise ValueError("T-based p-value rows must use df=n in the lookup table.")
    if not np.allclose(f_df2, results.loc[ancova, "n"]):
        raise ValueError("ANCOVA p-value rows must use df2=n in the lookup table.")

    # The lookup stores the native statistic for each test: t, r, and F.
    p_values.loc[ttest] = pvalue_ttest(
        statistic.loc[ttest], results.loc[ttest, "df"]
    )
    p_values.loc[partial] = pvalue_partial_corr(
        statistic.loc[partial], results.loc[partial, "df"]
    )
    p_values.loc[ancova] = pvalue_ancova(
        statistic.loc[ancova], results.loc[ancova, "df2"]
    )
    if p_values.isna().any():
        raise ValueError("Could not compute p-values for every lookup row.")
    return p_values


def prepare_figure4_data(lookup_path=DEFAULT_LOOKUP_PATH):
    """Compute p-values from lookup parameters and calculate flip probabilities.

    Each output row retains the lookup metadata (Threshold, Metric, NPVR, n,
    formula, statistic, df and df2), so its ``sigma_p`` is exactly the value
    produced by the matching Table 1 formula rather than a shared uncertainty.
    """
    results = load_sigma_p_lookup(lookup_path).reset_index(drop=True)
    results["p_value"] = compute_lookup_pvalues(results)
    results["p_minus_alpha"] = results["p_value"] - ALPHA
    results["flip_probability"] = flip_proba_beta(results["p_value"], results["sigma_p"])
    results["significance"] = np.where(
        results["p_value"] <= ALPHA, "significant", "non-significant"
    )
    return results


def _binned_flip_statistics(results, left, right, width, include_left):
    """Return bin centres, means, and standard errors for flip probabilities.

    The standard error is shown only where a bin has at least four points, so
    the shaded band does not imply precision from very small bins.
    """
    values = results["p_minus_alpha"]
    mask = ((values >= left) if include_left else (values > left)) & (values <= right)
    subset = results.loc[mask, ["p_minus_alpha", "flip_probability"]].copy()
    if subset.empty:
        return pd.DataFrame(columns=["center", "mean", "se", "count"])

    edges = np.arange(left, right + width, width)
    if edges[-1] < right:
        edges = np.append(edges, right)
    subset["bin"] = pd.cut(subset["p_minus_alpha"], edges, include_lowest=include_left)
    stats = subset.groupby("bin", observed=True)["flip_probability"].agg(["mean", "std", "count"])
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats.loc[stats["count"] <= 3, "se"] = np.nan
    stats["center"] = [interval.mid for interval in stats.index]
    return stats.reset_index(drop=True)


def plot_figure4(results, output_path=DEFAULT_OUTPUT_PATH, alpha=ALPHA, cohort_results=None):
    """Save Figure 4 with optional HC and PD mean-flip comparison lines."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting requires matplotlib. Install it in the same Python environment "
            "that provides pandas and scipy."
        ) from error

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {"formula", "p_minus_alpha", "flip_probability"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    cohort_results = cohort_results or {}
    for cohort, cohort_data in cohort_results.items():
        missing = {"p_minus_alpha", "flip_probability"}.difference(cohort_data.columns)
        if missing:
            raise ValueError(f"{cohort} results are missing columns: {sorted(missing)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.18, wspace=0.08)
    formulas = (
        "two_sample_t_p_value",
        "partial_correlation_p_value",
        "ancova_p_value",
    )
    # Use equal p-alpha intervals and the same 0.005 bin width on both sides
    # of alpha, so the two panels are directly comparable.
    symmetric_limit = min(alpha, float(results["p_minus_alpha"].max()))
    negative = _binned_flip_statistics(
        results, -symmetric_limit, 0, 0.005, include_left=True
    )
    positive = _binned_flip_statistics(
        results, 0, symmetric_limit, 0.005, include_left=False
    )
    cohort_styles = {
        "HC": {
            "color": "#0072B2", "label": "HC mean",
            "se_label": "HC standard error",
        },
        "PD": {
            "color": "#D62728", "label": "PD mean",
            "se_label": "PD standard error",
        },
    }
    panel_specs = (
        (axes[0], results["p_minus_alpha"] <= 0, negative,
         "False-positive risk", (-symmetric_limit, 0)),
        (axes[1], results["p_minus_alpha"] > 0, positive,
         "False-negative risk", (0, symmetric_limit)),
    )
    for ax, mask, stats, title, xlim in panel_specs:
        for formula in formulas:
            group = results.loc[mask & (results["formula"] == formula)]
            if group.empty:
                continue
            ax.scatter(
                group["p_minus_alpha"], group["flip_probability"],
                s=12, alpha=0.38, color="#000000", linewidths=0,
            )
        if stats.empty:
            continue
        ax.plot(stats["center"], stats["mean"], "o-", color="#000000", alpha=0.60,
                ms=4, lw=1.5, label="PD+HC Mean", zorder=5)
        valid_se = stats["se"].notna()
        ax.fill_between(
            stats.loc[valid_se, "center"],
            stats.loc[valid_se, "mean"] - stats.loc[valid_se, "se"],
            stats.loc[valid_se, "mean"] + stats.loc[valid_se, "se"],
            color="#808080", alpha=0.18,
            label="Standard error",
            zorder=2,
        )
        for cohort, style in cohort_styles.items():
            cohort_data = cohort_results.get(cohort)
            if cohort_data is None:
                continue
            cohort_stats = _binned_flip_statistics(
                cohort_data, xlim[0], xlim[1], 0.005, include_left=(xlim[0] < 0)
            )
            if cohort_stats.empty:
                continue
            valid_cohort_se = cohort_stats["se"].notna()
            ax.fill_between(
                cohort_stats.loc[valid_cohort_se, "center"],
                cohort_stats.loc[valid_cohort_se, "mean"] - cohort_stats.loc[valid_cohort_se, "se"],
                cohort_stats.loc[valid_cohort_se, "mean"] + cohort_stats.loc[valid_cohort_se, "se"],
                color=style["color"], alpha=0.16, label=style["se_label"], zorder=3,
            )
            ax.plot(
                cohort_stats["center"], cohort_stats["mean"], "o-",
                color=style["color"], ms=4, lw=1.5, label=style["label"], zorder=4,
            )
        ax.set(
            title=title,
            xlabel=r"Distance to significance threshold $p - \alpha$",
            xlim=xlim,
            ylim=(-0.02, 1.02),
        )
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Probability of significance flip")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper right",
               bbox_to_anchor=(0.90, 0.92))
    fig.text(
        0.5, 0.02,
        "Probability of numerically induced significant flips.",
        ha="center", va="bottom", fontsize=10,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def mean_flip_probability_near_alpha(results, half_width=0.04):
    """Return the mean flip probability for |p-alpha| <= ``half_width``."""
    if half_width < 0:
        raise ValueError("half_width must be non-negative.")
    subset = results.loc[results["p_minus_alpha"].between(-half_width, half_width)]
    if subset.empty:
        return np.nan, 0
    return float(subset["flip_probability"].mean()), len(subset)


def prepare_flip_table(results, min_flip_probability=0.0):
    """Return rows for which numerical uncertainty can change significance.

    ``flip_probability`` is a Beta-tail probability, not the outcome of one
    re-run.  Consequently, a row is included when this probability is above
    ``min_flip_probability`` (strictly greater, so zero-probability rows are
    excluded).  The direction labels describe the *possible* changed result.
    """
    required = {
        "Threshold", "Metric", "NPVR", "n", "formula", "p_value",
        "sigma_p", "flip_probability", "significance",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    if min_flip_probability < 0 or min_flip_probability >= 1:
        raise ValueError("min_flip_probability must be in the interval [0, 1).")

    flips = results.loc[
        results["flip_probability"] > min_flip_probability
    ].copy()
    # Keep the distance to the significance cutoff explicit in both the CSV
    # and the reviewer PDF, even when the input results file predates it.
    flips["p_minus_alpha"] = flips["p_value"] - ALPHA
    flips["possible_numerical_outcome"] = np.where(
        flips["significance"].eq("significant"),
        "non-significant",
        "significant",
    )
    flips["flip_direction"] = np.where(
        flips["significance"].eq("significant"),
        "significant -> non-significant",
        "non-significant -> significant",
    )
    flips["risk_type"] = np.where(
        flips["significance"].eq("significant"),
        "false-positive risk",
        "false-negative risk",
    )
    columns = [
        "Metric", "statistic", "n", "Threshold", "p_value", "p_minus_alpha",
        "NPVR", "formula", "df", "df2", "sigma_p", "flip_probability", "significance",
        "possible_numerical_outcome", "flip_direction", "risk_type",
    ]
    columns = [column for column in columns if column in flips.columns]
    return flips.loc[:, columns].sort_values(
        ["Metric", "risk_type", "flip_probability"],
        ascending=[True, True, False], kind="stable"
    ).reset_index(drop=True)


def generate_all_population_outputs(
    min_flip_probability=0.0,
):
    """Generate CSV results and filtered tables for combined, PD, and HC."""
    generated_paths = []

    for population, lookup_path in POPULATION_LOOKUP_PATHS.items():
        output_dir = TABLES_DIR / population
        output_dir.mkdir(parents=True, exist_ok=True)

        results = prepare_figure4_data(lookup_path)
        flips = prepare_flip_table(results, min_flip_probability)

        results_path = output_dir / "p_value_flip_results.csv"
        flip_csv_path = output_dir / "significance_flip_probability.csv"
        results.to_csv(results_path, index=False)
        flips.to_csv(flip_csv_path, index=False)

        generated_paths.extend((results_path, flip_csv_path))
        print(
            f"Generated {population}: {len(results):,} probability rows and "
            f"{len(flips):,} possible-flip rows."
        )

    return generated_paths


def build_parser():
    """Create the command-line interface in one testable place."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=ReproducibleHelpFormatter,
    )
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS_PATH,
        help="Input results CSV containing p_value, sigma_p, and flip_probability.",
    )
    parser.add_argument(
        "--lookup", type=Path, default=DEFAULT_LOOKUP_PATH,
        help="Expanded table1_all_formulae_lookup_values.csv used when --recompute-results is selected.",
    )
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_FLIP_TABLE_CSV)
    parser.add_argument("--skip-flip-csv", action="store_true", help="Do not write the filtered significance-flip CSV.")
    parser.add_argument(
        "--min-flip-probability", type=float, default=0.0,
        help="Only include rows whose flip probability is strictly greater than this value.",
    )
    parser.add_argument(
        "--recompute-results", action="store_true",
        help="Recompute p-values and flip probabilities from the selected expanded lookup before filtering.",
    )
    parser.add_argument(
        "--all-populations", action="store_true",
        help="Generate combined, PD, and HC CSV outputs from their matching expanded lookups.",
    )
    return parser


def main():
    """Generate CSV significance-flip outputs."""
    args = build_parser().parse_args()

    if args.all_populations:
        generate_all_population_outputs(
            min_flip_probability=args.min_flip_probability,
        )
        return

    if args.recompute_results:
        results = prepare_figure4_data(args.lookup)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.results, index=False)
        print(f"Recomputed {len(results):,} rows and saved them to {args.results}")
    else:
        results = pd.read_csv(args.results)
    mean_flip, near_alpha_count = mean_flip_probability_near_alpha(results, half_width=0.04)
    print(
        "Average flip probability for p - alpha in [-0.04, 0.04] "
        f"(n={near_alpha_count:,}): {mean_flip:.6f}"
    )
    if not args.skip_flip_csv:
        flips = prepare_flip_table(results, args.min_flip_probability)
    if not args.skip_flip_csv:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        flips.to_csv(args.csv_output, index=False)
        print(f"Saved {len(flips):,} possible flip rows to {args.csv_output}")


if __name__ == "__main__":
    main()
