# Figure and lookup-table reproducibility

This directory contains the preprocessing heuristic, connectome and graph
analysis code, and plotting notebooks used to generate Figures 1–6 and related
supplementary outputs for the numerical-variability analysis of functional MRI
graph measures. The plotting notebooks preserve the layout, colors, labels,
and captions used in the manuscript and export PDF figures directly from
Matplotlib.

> **Data availability:** the participant-level and intermediate DataFrames used
> for Figures 2–6 contain data obtained from the Parkinson's Progression
> Markers Initiative (PPMI). These files are not distributed in this GitHub
> repository. Code and publication figures can be shared, but restricted input
> tables, pickles, cached notebook outputs, and subject-level results must remain
> outside version control.

The Figure 2–4 notebooks load their restricted quality-control exclusion lists
from the path in `PPMI_SUBJECT_EXCLUSIONS_FILE`, or from the ignored local file
`data/subject_exclusions.json`. That JSON file must contain `pd` and `hc`
arrays and must never be committed.

## Contents

| File | Purpose |
|---|---|
| [`heuristic.py`](heuristic.py) | HeuDiConv heuristic used to map PPMI anatomical and resting-state functional DICOM series to a BIDS-compatible NIfTI organization. |
| [`Data_processingcopy_fix.py`](Data_processingcopy_fix.py) | Extracts Schaefer-atlas regional time series, constructs functional connectomes, computes NetworkX graph metrics, and writes private DataFrames to `Result_tableWConf_*.pkl` (with confound regression) and `Result_tableNoConf_*.pkl` (without confound regression). These files are consumed by the figure notebooks. |
| [`Fig1-simulation.ipynb`](Fig1-simulation/Fig1-simulation.ipynb) | Generates the four-panel conceptual simulation in Figure 1; it does not use participant-level PPMI measurements. Its PDFs are in [`Fig1-simulation/Figures/`](Fig1-simulation/Figures/). |
| [`Fig2and3-Graphmetrics_NPVR.ipynb`](Fig2-3-GraphMetrics-NPVR/Fig2and3-Graphmetrics_NPVR.ipynb) | Computes numerical variability (NV), population variability (PV), and NPVR; generates Figure 2 for local graph metrics and Figure 3 for global graph metrics. The PDF and PNG outputs are in [`Fig2-3-GraphMetrics-NPVR/`](Fig2-3-GraphMetrics-NPVR/). |
| [`Fig4-RegionalVariability.ipynb`](Fig4-RegionalVariability/Fig4-RegionalVariability.ipynb) | Generates the combined PD+HC regional NPVR map in [`Fig4-RegionalVariability/`](Fig4-RegionalVariability/) and the separate PD and HC supplementary maps in [`Fig4-RegionalVariability/Figures/`](Fig4-RegionalVariability/Figures/). |
| [`Fig5and6and-Conf_vs_NoConf.ipynb`](Fig5and6and-Conf_vs_NoConf/Fig5and6and-Conf_vs_NoConf.ipynb) | Evaluates the effect of confound regression and performs sign-flip permutation inference. Main figures are in [`Fig5and6and-Conf_vs_NoConf/Figures/`](Fig5and6and-Conf_vs_NoConf/Figures/), and population-specific and diagnostic outputs are in [`Figures/supplementaryFigures/`](Fig5and6and-Conf_vs_NoConf/Figures/supplementaryFigures/). |
| [`make_npvr_uncertainty_tables.py`](Unccertainity_lookuptable/make_npvr_uncertainty_tables.py) | Propagates NPVR through the manuscript uncertainty formulae and creates lookup tables for combined PD+HC, PD, and HC. |
| [`pValflip.py`](SignificanceFlipProb/pValflip.py) | Computes p-values and Beta-tail significance-flip probabilities, creates population-specific CSV tables, and generates the comparison figure in [`SignificanceFlipProb/Figures/`](SignificanceFlipProb/Figures/). |
| [`tables/`](tables/) | Generated uncertainty and significance-flip tables, separated into `combined/`, `PD/`, and `HC/`. |

`heuristic.py` and `Data_processingcopy_fix.py` belong to the upstream imaging
workflow. They are not required for Figure 1 or for the two lookup-table
commands described below.

### Repository layout

```text
.
├── Data_processingcopy_fix.py
├── heuristic.py
├── Fig1-simulation/
│   ├── Fig1-simulation.ipynb
│   └── Figures/
├── Fig2-3-GraphMetrics-NPVR/
│   ├── Fig2and3-Graphmetrics_NPVR.ipynb
│   └── Figure 2 and Figure 3 PDF/PNG files
├── Fig4-RegionalVariability/
│   ├── Fig4-RegionalVariability.ipynb
│   ├── fig4-PD+HC.pdf
│   └── Figures/                 # PD and HC supplementary maps
├── Fig5and6and-Conf_vs_NoConf/
│   ├── Fig5and6and-Conf_vs_NoConf.ipynb
│   └── Figures/
│       ├── main Figure 5–6 outputs
│       └── supplementaryFigures/
├── SignificanceFlipProb/
│   ├── pValflip.py
│   └── Figures/                 # significance-flip comparison PDF
├── Unccertainity_lookuptable/
│   ├── make_npvr_uncertainty_tables.py
│   ├── NPVR_table.csv
│   ├── NPVR_tablePD.csv
│   └── NPVR_tableHC.csv
└── tables/
    ├── combined/
    ├── PD/
    └── HC/
```

### Figure index

| Manuscript output | File or directory |
|---|---|
| Figure 1 components | [`sim1.pdf`](Fig1-simulation/Figures/sim1.pdf), [`sim2.pdf`](Fig1-simulation/Figures/sim2.pdf), and [`sim3.pdf`](Fig1-simulation/Figures/sim3.pdf) |
| Figure 2 — local graph metrics | [`Fig2_local_metrics.pdf`](Fig2-3-GraphMetrics-NPVR/Fig2_local_metrics.pdf) |
| Figure 3 — global graph metrics | [`Fig3_global_metrics.pdf`](Fig2-3-GraphMetrics-NPVR/Fig3_global_metrics.pdf) |
| Figure 4 — combined PD+HC regional NPVR | [`fig4-PD+HC.pdf`](Fig4-RegionalVariability/fig4-PD+HC.pdf) |
| Figure 4 — supplementary PD and HC maps | [`Fig4-RegionalVariability/Figures/`](Fig4-RegionalVariability/Figures/) |
| Figures 5–6 — combined PD+HC outputs | [`Fig5and6and-Conf_vs_NoConf/Figures/`](Fig5and6and-Conf_vs_NoConf/Figures/) |
| Figures 5–6 — population-specific and diagnostic outputs | [`supplementaryFigures/`](Fig5and6and-Conf_vs_NoConf/Figures/supplementaryFigures/) |
| Figure 7 — significance-flip probability | [`fig7_flip_probability.pdf`](SignificanceFlipProb/Figures/fig7_flip_probability.pdf) |

## End-to-end processing provenance

The study used the following processing sequence:

```text
PPMI DICOM images
        │
        ▼
HeuDiConv 1.2 + heuristic.py
DICOM-to-NIfTI conversion and BIDS organization
        │
        ▼
fMRIPrep 23.2.1 + Verificarlo fuzzy-libmath
anatomical/functional preprocessing and floating-point perturbations
        │
        ▼
Data_processingcopy_fix.py + Python 3.12 + NetworkX 3.5
regional time series, connectomes, graph metrics, Result_table*.pkl
        │
        ▼
Fig2-3-GraphMetrics-NPVR/Fig2and3-Graphmetrics_NPVR.ipynb,
Fig4-RegionalVariability/Fig4-RegionalVariability.ipynb, and
Fig5and6and-Conf_vs_NoConf/Fig5and6and-Conf_vs_NoConf.ipynb + Python 3.11
NV, PV, NPVR, statistical analyses, and Figures 2–6
```

### DICOM conversion and preprocessing

All anatomical and functional MRI data were preprocessed with **fMRIPrep
23.2.1**. Before preprocessing, the original DICOM images were converted to
NIfTI and organized for fMRIPrep using **HeuDiConv 1.2** and the repository's
[`heuristic.py`](heuristic.py). The heuristic maps the heterogeneous PPMI
anatomical and resting-state fMRI series descriptions to BIDS-compatible T1w
and BOLD filenames.

DICOM conversion and fMRIPrep were run on **Narval**, a high-performance
computing cluster hosted by Calcul Québec and operated as part of the Digital
Research Alliance of Canada.

### Numerical perturbation

Floating-point perturbations were introduced with the **fuzzy-libmath**
instrumentation provided by the Verificarlo framework. The fMRIPrep 23.2.1
workflow was instrumented with fuzzy-libmath and executed repeatedly on Narval
to characterize numerical variability. The framework documentation and public
container information are available in the
[Verificarlo fuzzy-libmath repository](https://github.com/verificarlo/fuzzy/tree/master#fuzzy-libmath).

### Connectomes and graph measures

Functional connectomes and graph measures were generated on **Rorqual**, also
hosted by Calcul Québec, using **Python 3.12** and **NetworkX 3.5**. The
[`Data_processingcopy_fix.py`](Data_processingcopy_fix.py) workflow:

1. loads each preprocessed BOLD image and its confounds table;
2. extracts regional time series using the Schaefer 2018 atlas with 100 regions
   and seven networks;
3. computes a Pearson-correlation functional connectivity matrix;
4. builds an undirected graph at correlation thresholds 0.05, 0.10, 0.20,
   0.30, 0.40, and 0.50;
5. computes degree centrality, betweenness centrality, eigenvector centrality,
   clustering coefficient, average shortest-path length, and small-worldness;
6. stores the connectivity matrices, adjacency matrices, graph measures, and
   acquisition metadata in population/batch DataFrames such as
   `Result_tableWConf_batch_1.pkl` through
   `Result_tableWConf_batch_hc.pkl`.

The `WConf` outputs include regression of the six rigid-body motion parameters
(`trans_x`, `trans_y`, `trans_z`, `rot_x`, `rot_y`, and `rot_z`); corresponding
`NoConf` outputs omit this regression. These pickle files contain
PPMI-derived records and are therefore private analysis inputs, not repository
artifacts.

#### Format of the `Result_table*.pkl` inputs

Each `Result_tableWConf_*.pkl` or `Result_tableNoConf_*.pkl` file is a pickled
pandas DataFrame. These files are the initial inputs to the downstream figure
notebooks. Each DataFrame row represents the graph-analysis result for one
specific combination of:

- subject identifier (`subject`);
- imaging session (`session`);
- fuzzy-libmath/MCA repetition (`repetition`); and
- resting-state acquisition or phase-encoding direction (`acquisition`).

The other columns contain the connectome and graph measurements calculated for
that exact observation:

| Column | Stored value |
|---|---|
| `degree_centralities` | Dictionary mapping each of the 100 Schaefer ROI indices to its degree-centrality value. |
| `betweenness_centralities` | Dictionary mapping ROI indices to betweenness-centrality values. |
| `eigenvector_centralities` | Dictionary mapping ROI indices to eigenvector-centrality values. |
| `clustering_coefficients` | Dictionary mapping ROI indices to clustering-coefficient values. |
| `avg_shortest_path_length` | One whole-graph average shortest-path-length value. |
| `small_worldness` | One whole-graph small-worldness value. |
| `correlation_matrix` | The 100 × 100 functional-correlation matrix before graph thresholding. |
| `adj_G` | The thresholded, weighted 100 × 100 graph adjacency matrix. |

The four regional metrics are stored as Python dictionaries whose keys are
Schaefer ROI indices and whose values are the corresponding regional metric
values. The two global metrics are stored as scalar values, while the
connectivity and adjacency data are stored as NumPy-style matrices. The figure
notebooks load these DataFrames and reorganize them internally by network
threshold and population as required by each analysis.

Although the subject identifiers are pseudonymous, they are derived from PPMI
records. Therefore, the pickle files, individual rows, screenshots, and cached
notebook displays of these DataFrames are not included in the public
repository.

### Downstream analysis

Downstream variability estimation, statistical analysis, table generation,
and figure production were performed locally on **Ubuntu 22.04.5 LTS** using
**Python 3.11**. In particular,
[`Fig2and3-Graphmetrics_NPVR.ipynb`](Fig2-3-GraphMetrics-NPVR/Fig2and3-Graphmetrics_NPVR.ipynb)
reads the previously generated
`Result_tableWConf_batch_*.pkl` files, applies the documented quality-control
exclusions, and calculates NV, PV, and NPVR. It does not construct the original
connectomes; that upstream step is implemented in `Data_processingcopy_fix.py`.

#### NV, PV, and subject-level NPVR tables

Using the repeated graph-metric measurements in the `Result_table*.pkl`
DataFrames, the downstream workflow estimates:

- **numerical variability (NV):** within-subject variability across the 10
  fuzzy-libmath/MCA repetitions;
- **population variability (PV):** between-subject variability for the same
  graph metric, acquisition, and network threshold; and
- **NPVR:** the Numerical-Population Variability Ratio, calculated as
  $\mathrm{NV}/\mathrm{PV}$.

[`Fig2and3-Graphmetrics_NPVR.ipynb`](Fig2-3-GraphMetrics-NPVR/Fig2and3-Graphmetrics_NPVR.ipynb)
calculates the combined-population NV, PV,
and NPVR directly from the quality-controlled pickle inputs. Subject-level
NPVR-derived results for the separate populations and thresholds are stored in
the private `Allpop/` CSV files. For example:

```text
dfW_stat05.csv    # PD, WConf pipeline, threshold 0.05
dfWhc_stat05.csv  # HC, WConf pipeline, threshold 0.05
dfN_stat05.csv    # PD, NoConf pipeline, threshold 0.05
dfNhc_stat05.csv  # HC, NoConf pipeline, threshold 0.05
```

The suffixes `05`, `1`, `2`, `3`, `4`, and `5` correspond to network
thresholds 0.05, 0.10, 0.20, 0.30, 0.40, and 0.50, respectively. Each row
retains the pseudonymous subject and relevant session/acquisition metadata,
together with the subject-level graph-metric NPVR terms:

| CSV column | NPVR representation |
|---|---|
| `degree` | Serialized array containing one value for each of the 100 Schaefer ROIs. |
| `betweeness` | Serialized 100-ROI array for betweenness centrality. The spelling follows the source files. |
| `eigenvec` | Serialized 100-ROI array for eigenvector centrality. |
| `clusteringcoef` | Serialized 100-ROI array for clustering coefficient. |
| `smallworldness` | One global whole-graph value per subject row. |
| `avg_shortestPathLength` | One global whole-graph value per subject row. |

The regional columns therefore preserve spatial variation across all 100
atlas regions, whereas the two global graph metrics contribute one scalar per
subject row. In the current analysis code, these CSV columns contain squared
subject-level NPVR terms. Population-specific NPVR is reconstructed as
$\sqrt{|\operatorname{mean}(x)|}$: local terms are averaged elementwise across
subjects to retain 100 regional values, while global terms are averaged to one
value. Figures report the regional distributions or their regional means as
appropriate.

These CSV files contain PPMI-derived subject identifiers and measurements and
are not distributed in the public repository.

The study code and notebooks are maintained at
[Numerical-Variability-of-functional-MRI-Graph-Measures](https://github.com/mina94az/Numerical-Variability-of-functional-MRI-Graph-Measures).

## Software environment

Python 3.11 or newer is recommended. Create an isolated environment from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "numpy>=2" \
  "pandas>=2" \
  "matplotlib>=3.7" \
  "scipy>=1.11" \
  "nilearn>=0.10" \
  "networkx==3.5" \
  nibabel \
  joblib \
  seaborn \
  plotly \
  jupyterlab
```

[`Fig4-RegionalVariability.ipynb`](Fig4-RegionalVariability/Fig4-RegionalVariability.ipynb)
uses Nilearn's 100-region, seven-network Schaefer
atlas. Nilearn may download the atlas the first time the notebook is run.

## Restricted data and local setup

Qualified researchers can apply for PPMI access through the official
[PPMI data-access page](https://www.ppmi-info.org/access-data-specimens/download-data).
New users must follow the current PPMI application process, including the Data
Use Agreement and publication policy. Access authorization does not transfer
through this repository; each researcher is responsible for complying with
the applicable PPMI and institutional requirements.

After obtaining authorized access and preparing the analysis inputs, set
`FMRI_DATA_ROOT` to an absolute local path:

```bash
export FMRI_DATA_ROOT=/absolute/path/to/private/fmri-analysis-data
jupyter lab
```

The notebooks expect the following private layout:

```text
$FMRI_DATA_ROOT/
├── table_correther0.05/
│   ├── Result_tableWConf_batch_1.pkl
│   ├── ...
│   └── Result_tableWConf_batch_hc.pkl
├── table_correther0.1/
├── table_correther0.2/
├── table_correther0.3/
├── table_correther0.4/
├── table_correther0.5/
└── Allpop/
    ├── dfW_stat05.csv
    ├── dfWhc_stat05.csv
    └── ...
```

The data directory can be located anywhere outside the repository. Do not copy
restricted inputs into a Git-tracked folder.

## Reproducing Figure 1

[`Fig1-simulation.ipynb`](Fig1-simulation/Fig1-simulation.ipynb) generates a
conceptual simulation rather than analyzing PPMI
participant measurements. Its four panels show:

- **A:** simulated measurement distributions for two populations with 10
  subjects each under low within-individual numerical variability;
- **B:** the corresponding distributions under high within-individual
  numerical variability;
- **C:** the Numerical-Population Variability Ratio
  ($\nu_{\mathrm{npv}}$) computed for the two simulated populations; and
- **D:** the expected variability of Cohen's $d$ ($\sigma_d$) as a function of
  sample size and $\nu_{\mathrm{npv}}$.

Run it independently with Jupyter:

```bash
jupyter lab Fig1-simulation/Fig1-simulation.ipynb
```

The notebook exports its component figures as
[`sim1.pdf`](Fig1-simulation/Figures/sim1.pdf),
[`sim2.pdf`](Fig1-simulation/Figures/sim2.pdf), and
[`sim3.pdf`](Fig1-simulation/Figures/sim3.pdf). Because the values are
simulated within the notebook, no PPMI input directory is required. Review the
simulation parameters and random-number initialization in the notebook before
interpreting exact values as reproducible numerical outputs.

## Reproducing Figures 2–6

Start Jupyter after setting `FMRI_DATA_ROOT`, then run the notebooks in this
order:

1. [`Fig2and3-Graphmetrics_NPVR.ipynb`](Fig2-3-GraphMetrics-NPVR/Fig2and3-Graphmetrics_NPVR.ipynb)
2. [`Fig4-RegionalVariability.ipynb`](Fig4-RegionalVariability/Fig4-RegionalVariability.ipynb)
3. [`Fig5and6and-Conf_vs_NoConf.ipynb`](Fig5and6and-Conf_vs_NoConf/Fig5and6and-Conf_vs_NoConf.ipynb)

Each notebook documents its calculations and input assumptions. Its figures
are stored beside the corresponding code:

- Figures 1A–D: [`Fig1-simulation/Figures/`](Fig1-simulation/Figures/)
- Figures 2–3: [`Fig2-3-GraphMetrics-NPVR/`](Fig2-3-GraphMetrics-NPVR/)
- Figure 4: [`Fig4-RegionalVariability/`](Fig4-RegionalVariability/)
- Figure 4 supplementary PD/HC maps:
  [`Fig4-RegionalVariability/Figures/`](Fig4-RegionalVariability/Figures/)
- Main Figures 5–6:
  [`Fig5and6and-Conf_vs_NoConf/Figures/`](Fig5and6and-Conf_vs_NoConf/Figures/)
- Figures 5–6 supplementary and permutation-diagnostic outputs:
  [`Fig5and6and-Conf_vs_NoConf/Figures/supplementaryFigures/`](Fig5and6and-Conf_vs_NoConf/Figures/supplementaryFigures/)

PDFs are generated directly by the plotting code, and PNG versions are
provided where available for GitHub previews.

The Figure 5–6 permutation analysis uses a fixed random seed so repeated runs
with the same software, inputs, and parameters reproduce the reported result.
The uncertainty and p-value-flip scripts are deterministic and do not draw
random samples.

Because the restricted DataFrames are not included, a public clone cannot
regenerate Figures 2–6 without separately authorized PPMI access and locally
prepared inputs. The committed PDFs and PNGs provide viewable manuscript
outputs without distributing their underlying participant-level tables.

## Reproducing uncertainty lookup tables

The lookup workflow requires these three local summary inputs:

```text
NPVR_table.csv
NPVR_tablePD.csv
NPVR_tableHC.csv
```

Each file must contain one row per graph metric and threshold with the columns
`Threshold`, `Metric`, and `NPVR`. These CSV files are ignored by Git. Confirm
that sharing any derived summary complies with the applicable data-use terms
before publishing it.

From the repository root, generate all three populations with:

```bash
python Unccertainity_lookuptable/make_npvr_uncertainty_tables.py
```

This creates:

```text
tables/
├── combined/
├── PD/
└── HC/
```

Each directory contains the complete numerical lookup CSV. Run
`python Unccertainity_lookuptable/make_npvr_uncertainty_tables.py --help` for
alternate sample sizes, statistic grids, and single-population input.

## Reproducing significance-flip tables and the comparison figure

After generating the uncertainty tables, run:

```bash
python SignificanceFlipProb/pValflip.py --all-populations
```

For each population, this produces:

- `p_value_flip_results.csv`: every evaluated configuration, including rows
  with zero or negligible flip probability;
- `significance_flip_probability.csv`: rows with
  `flip_probability > min_flip_probability`, plus interpretation columns for
  flip direction and risk type.

`pValflip.py` does not generate PDF lookup tables. Its only PDF output in the
three-population workflow is the comparison figure below.

The comparison figure is generated directly as:

[`SignificanceFlipProb/Figures/fig7_flip_probability.pdf`](SignificanceFlipProb/Figures/fig7_flip_probability.pdf)

It shows combined PD+HC in black, PD in red, and HC in blue. Shaded regions are
standard errors. The left panel shows false-positive risk and the right panel
shows false-negative risk.

To apply a stricter reporting cutoff, for example 10%, use:

```bash
python SignificanceFlipProb/pValflip.py \
  --all-populations \
  --min-flip-probability 0.10
```

## Before publishing to GitHub

The repository `.gitignore` excludes CSV, pickle, NIfTI, HDF5, Jupyter
checkpoint, and Python cache files. Still perform a manual review before every
push:

1. Clear notebook outputs because `.ipynb` files can embed displayed
   DataFrames, absolute paths, and other restricted information:

   ```bash
   jupyter nbconvert \
     --ClearOutputPreprocessor.enabled=True \
     --inplace \
     Fig1-simulation/Fig1-simulation.ipynb \
     Fig2-3-GraphMetrics-NPVR/Fig2and3-Graphmetrics_NPVR.ipynb \
     Fig4-RegionalVariability/Fig4-RegionalVariability.ipynb \
     Fig5and6and-Conf_vs_NoConf/Fig5and6and-Conf_vs_NoConf.ipynb
   ```

2. Inspect everything that will be committed:

   ```bash
   git status --short
   git diff --cached --name-only
   ```

3. Do not use `git add -f` for ignored data files. Check that no subject IDs,
   private paths, CSV files, pickles, or intermediate DataFrames are staged.

4. Commit code, this README, and only those manuscript figures or aggregate
   outputs that you have confirmed may be shared.

## Data-availability statement

The following wording can be adapted for the repository or manuscript:

> The analysis code and publication figures are available in this repository.
> Participant-level and intermediate data used to generate Figures 2–6 were
> obtained from the Parkinson's Progression Markers Initiative (PPMI) and are
> not redistributed here. Qualified researchers may request access directly
> from PPMI and must comply with the applicable Data Use Agreement and
> publication policy. After obtaining authorized access, users can reproduce
> the analyses by preparing the documented local input structure and setting
> the `FMRI_DATA_ROOT` environment variable.

## Citation

If you reuse this workflow, cite the associated manuscript and acknowledge
PPMI according to its current publication policy. Add the final manuscript
citation here when it becomes available.
