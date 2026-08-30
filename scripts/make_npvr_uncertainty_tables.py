#!/usr/bin/env python3
"""Generate reproducible NPVR uncertainty lookup tables.

The script reads an NPVR summary CSV and propagates each value through the
uncertainty expressions used in the manuscript. It writes the expanded
machine-readable ``table1_all_formulae_lookup_values.csv`` file and the
LaTeX formula and lookup-table fragments.

Run from any directory; all default paths are resolved relative to this file.
With no arguments, the combined, PD, and HC tables are generated together::

    python Unccertainity_lookuptable/make_npvr_uncertainty_tables.py

The outputs are written to ``tables/{combined,PD,HC}``. To
generate only one population, select its input and output directory::

    python Unccertainity_lookuptable/make_npvr_uncertainty_tables.py \
        --input Unccertainity_lookuptable/NPVR_tablePD.csv \
        --output-dir tables/PD

The calculations are deterministic and do not use random sampling.  The
default sample sizes and statistic grids are declared below so the complete
analysis configuration is recorded in version control.

Implemented uncertainty expressions:

* Cohen's d: sigma_d ~= 2 nu_npv / sqrt(n)
* two-sample t: sigma_t ~= nu_npv; its p-value uncertainty is
  sigma_p >= 2 f_t,df(|t|) nu_npv, with t computed from r and n
* two-sample t p-value: sigma_p >= 2 f_t,df(|t|) (nu_npv)
* partial correlation: sigma_r >= nu_npv sqrt((1-r^2)/(n-1)), with df = n
* partial-correlation p-value: sigma_p >= 2 f_t,df(|t|) (nu_npv/(1-r^2)) sqrt(df / (n-1))
* ANCOVA: sigma_F ~= 2 sqrt(F) nu_npv
* ANCOVA p-value: sigma_p ~= 2 sqrt(F) f_F(F; 1, n) nu_npv

Pass the command-line options to regenerate tables for other statistics and
degrees of freedom.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "Unccertainity_lookuptable"

DEFAULT_SAMPLE_SIZES = (30, 50, 75, 100, 300, 500, 1000)
DEFAULT_R_VALUES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
DEFAULT_T_VALUES = (0.5, 1, 2, 3, 4, 5, 6)
DEFAULT_F_VALUES = (0.5, 1, 2, 4, 6, 8, 10, 15)
DEFAULT_INPUTS = {
    "combined": INPUT_DIR / "NPVR_table.csv",
    "PD": INPUT_DIR / "NPVR_tablePD.csv",
    "HC": INPUT_DIR / "NPVR_tableHC.csv",
}
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "tables"

METRIC_LABELS = {
    "degree": "Degree centrality",
    "betweeness": "Betweeness centrality",
    "eigenvec": "Eigenvector centrality",
    "clusteringcoef": "Clustering coefficient",
    "smallworldness": "Smallworldness",
    "avg_shortestPathLength": "Average shortest path",
}
METRIC_ORDER = list(METRIC_LABELS)

EXPORT_COLUMNS = [
    "Threshold", "Metric", "NPVR", "n", "formula", "df", "statistic",
    "df2", "p_value", "uncertainty",
]


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def parse_number_list(raw: str, kind: type[int] | type[float], name: str) -> list[int] | list[float]:
    try:
        values = [kind(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"at least one {name} is required")
    if any(value <= 0 for value in values) and name != "r values":
        raise argparse.ArgumentTypeError(f"{name} must be positive")
    return values


def parse_sample_sizes(raw: str) -> list[int]:
    values = parse_number_list(raw, int, "sample size")
    if any(value <= 3 for value in values):
        raise argparse.ArgumentTypeError("sample sizes must be greater than 3")
    return values  # type: ignore[return-value]


def parse_r_values(raw: str) -> list[float]:
    values = parse_number_list(raw, float, "r values")
    if any(abs(value) >= 1 for value in values):
        raise argparse.ArgumentTypeError("r values must lie strictly between -1 and 1")
    return values  # type: ignore[return-value]


def parse_f_values(raw: str) -> list[float]:
    return parse_number_list(raw, float, "F values")  # type: ignore[return-value]


def load_npvr_table(path: Path) -> pd.DataFrame:
    """Load and validate one NPVR value per metric and network threshold."""
    if not path.is_file():
        raise FileNotFoundError(
            f"NPVR input not found: {path}\n"
            "Use --input to select NPVR_table.csv, NPVR_tablePD.csv, or "
            "NPVR_tableHC.csv."
        )
    df = pd.read_csv(path)
    required = {"Threshold", "Metric", "NPVR"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    # Keep optional native statistic columns.  A supplied lookup table can
    # therefore define the actual t, r and F grids used in the study instead
    # of relying on the illustrative defaults below.
    statistic_columns = [column for column in ("t", "r", "F") if column in df.columns]
    df = df.loc[:, ["Threshold", "Metric", "NPVR", *statistic_columns]].copy()
    df = df.astype({"Threshold": float, "Metric": str, "NPVR": float})
    numeric_values = df[["Threshold", "NPVR"]].to_numpy().ravel()
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("Threshold and NPVR values must be finite numbers")
    if (df["Threshold"] <= 0).any():
        raise ValueError("Threshold values must be positive")
    if (df["NPVR"] < 0).any():
        raise ValueError("NPVR values must be non-negative")
    if df.duplicated(["Metric", "Threshold"]).any():
        raise ValueError("NPVR_table contains duplicate metric/threshold pairs")
    unknown_metrics = sorted(set(df["Metric"]) - set(METRIC_ORDER))
    missing_metrics = sorted(set(METRIC_ORDER) - set(df["Metric"]))
    if unknown_metrics or missing_metrics:
        raise ValueError(
            "NPVR metric names do not match the expected analysis metrics. "
            f"Missing: {missing_metrics or 'none'}; unknown: {unknown_metrics or 'none'}"
        )
    return df.sort_values(["Metric", "Threshold"]).reset_index(drop=True)


def statistic_values_from_lookup(
    npvr_table: pd.DataFrame, column: str, fallback: list[float]
) -> list[float]:
    """Use the distinct non-missing native lookup statistics when available."""
    if column not in npvr_table.columns:
        return fallback
    values = sorted(pd.to_numeric(npvr_table[column], errors="raise").dropna().unique().tolist())
    if not values:
        return fallback
    if column == "r" and any(abs(value) >= 1 for value in values):
        raise ValueError("lookup-table r values must lie strictly between -1 and 1")
    if column != "r" and any(value <= 0 for value in values):
        raise ValueError(f"lookup-table {column} values must be positive")
    return values


def expand_npvr(npvr_table: pd.DataFrame, parameter: str, values: list[float] | list[int]) -> pd.DataFrame:
    return npvr_table.merge(pd.DataFrame({parameter: values}), how="cross")


def compute_all_uncertainties(
    npvr_table: pd.DataFrame,
    sample_sizes: list[int],
    t_values: list[float],
    r_values: list[float],
    f_values: list[float],
) -> dict[str, pd.DataFrame]:
    by_n = expand_npvr(npvr_table, "n", sample_sizes)
    cohens_d = by_n.copy()
    cohens_d["uncertainty"] = cohens_d["NPVR"] * (2 / cohens_d["n"].map(math.sqrt))

    # A two-sample t p-value is evaluated from the stored t statistic.  Do
    # not derive t from r here: r belongs only to partial correlation.
    two_sample_t = expand_npvr(by_n, "statistic", t_values)
    # Two-sample t tables use the explicit t-value grid. The numerical
    # standard deviation itself is sigma_t ~= NPVR, while p-value uncertainty
    # is evaluated at each t and df=n.
    two_sample_t["df"] = two_sample_t["n"]
    two_sample_t["uncertainty"] = two_sample_t["NPVR"]

    t_p_value = two_sample_t.copy()
    t_p_value["statistic_type"] = "t"
    t_p_value["uncertainty"] = 2 * stats.t.pdf(
        t_p_value["statistic"].abs(), t_p_value["df"]
    ) * t_p_value["NPVR"]
    t_p_value["p_value"] = 2 * stats.t.sf(
        t_p_value["statistic"].abs(), t_p_value["df"]
    )

    partial = expand_npvr(by_n, "statistic", r_values)
    # The requested convention is df=n for the r-to-t conversion and df2=n
    # for F tests.  ``statistic`` is r in the t-based formulae and F in the
    # ANCOVA formulae.
    partial["df"] = partial["n"]
    one_minus_r2 = 1 - partial["statistic"] ** 2

    # Partial-correlation p-value uncertainty derives t from each sampled r
    # and n value: t = r sqrt(df / (1-r^2)).
    t_from_r = partial["statistic"] * (partial["df"] / one_minus_r2).map(math.sqrt)

    partial_r = partial.copy()
    partial_r["statistic_type"] = "r"
    partial_r["uncertainty"] = partial_r["NPVR"] * (
        one_minus_r2 / (partial_r["n"] - 1)
    ).map(math.sqrt)

    partial_p_value = partial.copy()
    partial_p_value["statistic_type"] = "r"
    partial_p_value["t_statistic"] = t_from_r
    partial_p_value["uncertainty"] = 2 * stats.t.pdf(
        t_from_r.abs(), partial_p_value["df"]
    ) * partial_p_value["NPVR"] * (
        partial_p_value["df"] / ((partial_p_value["n"] - 1) * one_minus_r2 ** 2)
    ).map(math.sqrt)
    partial_p_value["p_value"] = 2 * stats.t.sf(
        partial_p_value["t_statistic"].abs(), partial_p_value["df"]
    )

    # ANCOVA p-values depend on df2.  The requested convention df2=n means
    # that every F setting is evaluated at each sample size.
    ancova = expand_npvr(by_n, "statistic", f_values)
    ancova["statistic_type"] = "F"
    ancova["uncertainty"] = 2 * ancova["statistic"].map(math.sqrt) * ancova["NPVR"]

    ancova_p_value = ancova.copy()
    ancova_p_value["df2"] = ancova_p_value["n"]
    ancova_p_value["uncertainty"] = 2 * ancova_p_value["statistic"].map(math.sqrt) * stats.f.pdf(
        ancova_p_value["statistic"], 1, ancova_p_value["df2"]
    ) * ancova_p_value["NPVR"]
    ancova_p_value["p_value"] = stats.f.sf(
        ancova_p_value["statistic"], 1, ancova_p_value["df2"]
    )

    return {
        "cohens_d": cohens_d,
        "two_sample_t": two_sample_t,
        "two_sample_t_p_value": t_p_value,
        "partial_correlation": partial_r,
        "partial_correlation_p_value": partial_p_value,
        "ancova": ancova,
        "ancova_p_value": ancova_p_value,
    }


def format_number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def make_formula_summary_table() -> str:
    """Render the manuscript-style formula table shown before the lookups."""
    rows = [
        (r"Two-sample $t$", r"\mbox{$\sigma_t \approx \nu_{\mathrm{npv}}$}", r"\mbox{$\sigma_p \approx 2f_{t,df}(|t|)\nu_{\mathrm{npv}}$}"),
        (r"ANCOVA", r"\mbox{$\sigma_F \approx 2\sqrt{F}\nu_{\mathrm{npv}}$}", r"\mbox{$\sigma_p \approx 2\sqrt{F}f_F(F;1,df_2)\nu_{\mathrm{npv}}$}"),
        (r"Cohen's $d$", r"\mbox{$\sigma_d \approx \nu_{\mathrm{npv}}\frac{2}{\sqrt{n}}$}", r"--"),
        (r"\shortstack[l]{Partial\\correlation}", r"\mbox{$\sigma_r \geq \nu_{\mathrm{npv}}\sqrt{\frac{1-r^2}{n-1}}$}", r"\mbox{$\sigma_p \geq 2f_{t,df}(|t|)\frac{\nu_{\mathrm{npv}}}{1-r^2}\sqrt{\frac{df}{n-1}}$}"),
    ]
    body = "\n".join([
        r"\makebox[\linewidth][c]{%",
        # The column widths plus the two inter-column gaps add up to one
        # \linewidth, keeping the table aligned with the surrounding prose.
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.16\linewidth} "
        r">{\raggedright\arraybackslash}p{0.32\linewidth} "
        r">{\raggedright\arraybackslash}p{0.46\linewidth}@{}}",
        r"\toprule",
        r"\textbf{Statistic} & \textbf{Numerical standard deviation} & \textbf{Numerical p-value uncertainty} \\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
    ])
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\tiny",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.15}",
        body,
        r"\caption{First-order numerical variability of common statistical tests under Monte Carlo Arithmetic "
        r"perturbations. Cohen's $d$ formula assumes large and equal group sizes. "
        r"$f_{t,df}$ and $f_F(F;1,df_2)$ denote the probability density functions of the Student's "
        r"$t$-distribution with $df$ degrees of freedom and the $F$-distribution with $(1,df_2)$ degrees "
        r"of freedom, respectively. The $p$-value approximation for the partial correlation uses "
        r"$t=r\left(df/(1-r^2)\right)^{1/2}$~\cite{Yohan2025NPVR}.}",
        r"\label{tab:npvr-uncertainty-formulae}",
        r"\end{table}",
        "",
    ])


def make_combined_lookup_table(
    tables: dict[str, pd.DataFrame], table_kind: str, digits: int
) -> str:
    """Combine every graph metric into one threshold-column lookup table."""
    thresholds = sorted(tables["cohens_d"]["Threshold"].unique())
    for metric in METRIC_ORDER:
        metric_thresholds = sorted(
            tables["cohens_d"].loc[
                tables["cohens_d"]["Metric"].eq(metric), "Threshold"
            ].unique()
        )
        if metric_thresholds != thresholds:
            raise ValueError(f"Threshold columns differ for graph metric {metric!r}.")

    if table_kind == "numerical":
        specs = [
            (r"Two-sample $t$ ($\sigma_t$)", "two_sample_t", "t", True),
            (r"ANCOVA ($\sigma_F$)", "ancova", "F", True),
            (r"Cohen's $d$ ($\sigma_d$)", "cohens_d", None, False),
            (r"\shortstack[l]{Partial\\correlation ($\sigma_r$)}", "partial_correlation", "r", False),
        ]
        title = "Numerical standard deviation of all graph metrics"
        label = "tab:numerical-all-graph-metrics"
    elif table_kind == "p_value":
        specs = [
            (r"Two-sample $t$ ($\sigma_p$)", "two_sample_t_p_value", "t", False),
            (r"ANCOVA ($\sigma_p$)", "ancova_p_value", "F", False),
            (r"\shortstack[l]{Partial\\correlation ($\sigma_p$)}", "partial_correlation_p_value", "r", False),
        ]
        title = "Numerical p-value uncertainty of all graph metrics"
        label = "tab:p-value-all-graph-metrics"
    else:
        raise ValueError(f"Unknown table kind: {table_kind}")

    column_count = len(thresholds) + 4
    rows: list[str] = []
    # Keep page breaks deterministic so the first data row of every
    # continuation page can restore the graph metric, test and statistic.
    # Forty compact data rows fit safely below the repeated longtable header
    # in the landscape layout used by the supplemental document.
    rows_per_page = 40
    rows_on_page = 0
    for metric in METRIC_ORDER:
        metric_label = latex_escape(METRIC_LABELS[metric])
        npvr = (
            tables["cohens_d"].loc[tables["cohens_d"]["Metric"].eq(metric)]
            .drop_duplicates("Threshold").set_index("Threshold")["NPVR"]
        )
        metric_threshold_header = " & ".join(
            r"\shortstack{\textbf{$\tau=" + f"{t:.2f}" + r"$}\\"
            + r"\textbf{NPVR}\\" + format_number(npvr[t], digits) + r"}"
            for t in thresholds
        )
        metric_header_row = (
            r"\rowcolor[gray]{0.92} \multicolumn{4}{l}{\textbf{" + metric_label
            + r"}} & " + metric_threshold_header + r" \\"
        )
        # Do not leave a metric's NPVR banner alone at the bottom of a page.
        if rows_on_page >= rows_per_page - 1:
            rows.append(r"\pagebreak")
            rows_on_page = 0
        rows.append(metric_header_row)
        rows.append(r"\midrule")
        rows_on_page += 1

        for test_number, (test_name, table_key, statistic_name, n_independent) in enumerate(specs):
            data = tables[table_key].loc[tables[table_key]["Metric"].eq(metric)].copy()
            if n_independent:
                data = data.drop_duplicates(["statistic", "Threshold"])
                index_columns = ["statistic"]
            elif statistic_name is None:
                index_columns = ["n"]
            else:
                index_columns = ["statistic", "n"]
            pivot = data.pivot(index=index_columns, columns="Threshold", values="uncertainty")
            pivot_rows = list(pivot.iterrows())
            previous_statistic: float | None = None
            for row_number, (index, values) in enumerate(pivot_rows):
                if n_independent:
                    statistic_value, n = float(index), "--"
                elif statistic_name is None:
                    statistic_value, n = None, int(index)
                else:
                    statistic_value, n = float(index[0]), int(index[1])
                continues_on_new_page = rows_on_page >= rows_per_page
                if continues_on_new_page:
                    rows.append(r"\pagebreak")
                    rows_on_page = 0
                    # Restore the metric-specific threshold/NPVR header so a
                    # continuation page never mixes unlabeled NPVR and sigma
                    # values in the body of the table.
                    rows.append(metric_header_row)
                    rows.append(r"\midrule")
                    rows_on_page += 1

                starts_statistic_group = (
                    row_number == 0 or statistic_value != previous_statistic
                )
                starts_metric = test_number == 0 and row_number == 0
                starts_test = row_number == 0

                # Merge repeated labels visually.  Restore every identifying
                # label only in the first row of a continuation page.
                graph_metric = metric_label if starts_metric or continues_on_new_page else ""
                statistical_test = test_name if starts_test or continues_on_new_page else ""

                # The column heading already identifies these as statistics;
                # show only the value and print it once for each repeated
                # sample-size group.
                statistic = (
                    "" if statistic_name is None
                    else "$" + f"{statistic_value:g}" + "$"
                )
                if not starts_statistic_group and not continues_on_new_page:
                    statistic = ""
                cells = " & ".join(format_number(values[t], digits) for t in thresholds)
                rows.append(
                    graph_metric + " & " + statistical_test + " & " + statistic
                    + " & " + str(n) + " & " + cells + r" \\"
                )
                rows_on_page += 1
                previous_statistic = statistic_value
                next_index = pivot_rows[row_number + 1][0] if row_number + 1 < len(pivot_rows) else None
                next_statistic = (
                    float(next_index) if n_independent and next_index is not None
                    else float(next_index[0]) if isinstance(next_index, tuple)
                    else None
                )
                if row_number + 1 == len(pivot_rows):
                    is_last_test = table_key == specs[-1][1]
                    if is_last_test:
                        # Close the graph-metric block across the entire table.
                        rows.append(
                            r"\arrayrulecolor[rgb]{0.20,0.35,0.50}"
                            r"\hline\arrayrulecolor{black}"
                        )
                    else:
                        # Preserve the test boundary without crossing the
                        # visually merged Graph metric column.
                        rows.append(
                            r"\arrayrulecolor[rgb]{0.20,0.35,0.50}"
                            + rf"\cline{{2-{column_count}}}"
                            + r"\arrayrulecolor{black}"
                        )
                elif statistic_value != next_statistic:
                    group_size = sum(
                        1
                        for group_index, _ in pivot_rows
                        if (
                            float(group_index) if n_independent
                            else float(group_index[0]) if isinstance(group_index, tuple)
                            else None
                        ) == statistic_value
                    )
                    if group_size > 1:
                        # Separate repeated-statistic groups from the
                        # Statistic column through the final threshold.
                        rows.append(
                            r"\arrayrulecolor[rgb]{0.55,0.65,0.75}"
                            + rf"\cline{{3-{column_count}}}"
                            + r"\arrayrulecolor{black}"
                        )

    header = (
        r"\textbf{Graph metric} & \textbf{Statistical test} & \textbf{Statistic value} & "
        r"\textbf{$n$} & \multicolumn{" + str(len(thresholds))
        + r"}{c}{\textbf{Correlation thresholds}} \\"
    )
    numeric_column = r">{\raggedleft\arraybackslash}p{0.0695\linewidth}"
    column_specification = (
        r"p{0.14\linewidth} p{0.15\linewidth} p{0.12\linewidth} "
        r">{\raggedleft\arraybackslash}p{0.035\linewidth} "
        + numeric_column * len(thresholds)
    )
    return "\n".join((
        r"\begin{landscape}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{" + column_specification + "}",
        r"\caption{" + title + r".}\label{" + label + r"}\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\multicolumn{" + str(column_count)
        + r"}{c}{\small\textbf{Table \thetable\ (continued)}} \\",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot", *rows,
        r"\end{longtable}", r"\end{landscape}", "",
    ))


def threshold_header_item(threshold: float) -> str:
    return rf"\textbf{{$\tau={threshold:.2f}$}}"


def write_outputs(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    """Write the expanded lookup CSV and LaTeX table fragments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formula_path = output_dir / "statistical_uncertainty_formulae.tex"
    numerical_path = output_dir / "numerical_standard_deviation_lookup.tex"
    p_value_path = output_dir / "numerical_p_value_uncertainty_lookup.tex"
    formula_path.write_text(make_formula_summary_table(), encoding="utf-8")
    numerical_path.write_text(
        make_combined_lookup_table(tables, "numerical", digits=4),
        encoding="utf-8",
    )
    p_value_path.write_text(
        make_combined_lookup_table(tables, "p_value", digits=4),
        encoding="utf-8",
    )
    values_path = output_dir / "table1_all_formulae_lookup_values.csv"
    values = pd.concat([df.assign(formula=name) for name, df in tables.items()], ignore_index=True)
    values = values.drop(columns=["statistic_type", "t_statistic"], errors="ignore")
    # Preserve native lookup statistics in the exported CSV: t for the
    # two-sample test, r for partial correlation and F for ANCOVA.  The
    # derived partial-correlation t remains an internal calculation only.
    values = values.loc[:, EXPORT_COLUMNS]
    values.to_csv(values_path, index=False)
    return [formula_path, numerical_path, p_value_path, values_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate expanded NPVR uncertainty lookup CSVs and LaTeX tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path,
        help="NPVR summary CSV; omit to generate combined, PD, and HC tables",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="output directory, or parent directory when generating all populations",
    )
    parser.add_argument("--sample-sizes", type=parse_sample_sizes, default=list(DEFAULT_SAMPLE_SIZES), help="comma-separated n values")
    parser.add_argument("--t-values", type=parse_f_values, help="comma-separated t values; input-column values take precedence over defaults")
    parser.add_argument("--r-values", type=parse_r_values, help="comma-separated partial-correlation values")
    parser.add_argument("--f-values", type=parse_f_values, help="comma-separated ANCOVA F values")
    return parser


def generate_lookup_tables(args: argparse.Namespace, input_path: Path, output_dir: Path) -> list[Path]:
    """Generate every configured uncertainty table for one population."""
    npvr_table = load_npvr_table(input_path)
    t_values = args.t_values or statistic_values_from_lookup(
        npvr_table, "t", list(DEFAULT_T_VALUES)
    )
    r_values = args.r_values or statistic_values_from_lookup(
        npvr_table, "r", list(DEFAULT_R_VALUES)
    )
    f_values = args.f_values or statistic_values_from_lookup(
        npvr_table, "F", list(DEFAULT_F_VALUES)
    )
    tables = compute_all_uncertainties(npvr_table, args.sample_sizes, t_values, r_values, f_values)
    return write_outputs(tables, output_dir)


def main() -> None:
    args = build_parser().parse_args()
    if args.input is not None:
        jobs = [("selected population", args.input, args.output_dir)]
    else:
        jobs = [
            (population, input_path, args.output_dir / population)
            for population, input_path in DEFAULT_INPUTS.items()
        ]

    total_files = 0
    for population, input_path, output_dir in jobs:
        written = generate_lookup_tables(args, input_path, output_dir)
        total_files += len(written)
        print(f"Generated {population} lookup tables in {output_dir}:")
        for path in written:
            print(f"  {path.name}")
    print(f"Generated {total_files} file(s) for {len(jobs)} population(s).")


if __name__ == "__main__":
    main()
