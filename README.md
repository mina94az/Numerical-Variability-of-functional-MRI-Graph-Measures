# Numerical variability of functional MRI graph measures

This repository contains code for measuring how floating-point perturbations
affect functional-connectivity graph metrics. The workflow uses repeated
fMRIPrep executions instrumented with Verificarlo fuzzy-libmath and compares
within-subject numerical variability with between-subject variability.

## Main code

- `heuristic.py` maps PPMI DICOM series to a BIDS-compatible organization for
  HeuDiConv.
- `Data_processingcopy_fix.py` extracts Schaefer-100 regional time series,
  constructs correlation networks, and calculates graph metrics with NetworkX.
- The analysis notebooks calculate numerical variability (NV), population
  variability (PV), the numerical-to-population variability ratio (NPVR), and
  the effect of confound regression.
- `Unccertainity_lookuptable/make_npvr_uncertainty_tables.py` and
  `SignificanceFlipProb/pValflip.py` calculate statistical-uncertainty lookup
  tables and significance-flip probabilities.

## Processing summary

1. DICOM images are converted and organized with HeuDiConv.
2. fMRIPrep 23.2.1 is run 10 times with fuzzy-libmath perturbations.
3. A Schaefer atlas with 100 regions and seven networks is used to extract
   regional time series.
4. Pearson-correlation networks are constructed at thresholds 0.05, 0.10,
   0.20, 0.30, 0.40, and 0.50.
5. Degree, betweenness, eigenvector centrality, clustering, average shortest
   path length, and small-worldness are calculated.

## Dataset files

Participant-level inputs are derived from the Parkinson's Progression Markers
Initiative (PPMI) and are not included in this public repository.

The initial graph-measure datasets are pickled pandas DataFrames:

- `Result_tableWConf_batch_1.pkl` to `Result_tableWConf_batch_5.pkl`: Parkinson's
  disease (PD), with motion-confound regression.
- `Result_tableWConf_batch_hc.pkl`: healthy controls (HC), with motion-confound
  regression.
- `Result_tableNoConf_*.pkl`: corresponding datasets without motion-confound
  regression.

Each row represents one subject, session, acquisition, and MCA repetition. The
main columns are:

| Column | Description |
|---|---|
| `subject`, `session` | Participant and visit identifiers |
| `repetition` | One of the 10 perturbed executions |
| `acquisition` | Resting-state acquisition direction |
| `degree_centralities` | Degree centrality for 100 regions |
| `betweenness_centralities` | Betweenness centrality for 100 regions |
| `eigenvector_centralities` | Eigenvector centrality for 100 regions |
| `clustering_coefficients` | Clustering coefficient for 100 regions |
| `avg_shortest_path_length` | Whole-network scalar |
| `small_worldness` | Whole-network scalar |
| `correlation_matrix`, `adj_G` | Correlation and thresholded adjacency matrices |

### `dfW...` and `dfWhc...` files

The downstream CSV datasets use compact names:

| Example | Population | Processing |
|---|---|---|
| `dfW_stat05.csv` | PD | With confound regression |
| `dfWhc_stat05.csv` | HC | With confound regression |
| `dfN_stat05.csv` | PD | Without confound regression |
| `dfNhc_stat05.csv` | HC | Without confound regression |

Here, `W` means *with confounds*, `N` means *no confounds*, and `hc` identifies
healthy controls. A name without `hc` refers to the PD population. The suffixes
`05`, `1`, `2`, `3`, `4`, and `5` represent thresholds 0.05, 0.10, 0.20, 0.30,
0.40, and 0.50.

Each CSV row retains subject/session/acquisition information and NPVR terms for
the six graph metrics. Regional columns (`degree`, `betweeness`, `eigenvec`,
and `clusteringcoef`) contain serialized 100-region arrays. Global columns
(`smallworldness` and `avg_shortestPathLength`) contain one value per row. The
spelling `betweeness` is retained from the source datasets. These values are
squared subject-level NPVR terms; the group value is reconstructed as
`sqrt(abs(mean(values)))`, elementwise for regional arrays.

## Variability calculation

- **NV:** within-subject variability across the 10 perturbed repetitions.
- **PV:** between-subject variability for the same processing condition.
- **NPVR:** `NV / PV`.

PD and HC are processed separately when population-specific NPVR values are
required. An NPVR below 1 means population variability is larger than numerical
variability; an NPVR above 1 means numerical variability is larger.

## Data access

Researchers must request PPMI access from the
[official PPMI website](https://www.ppmi-info.org/access-data-specimens/download-data)
and follow its Data Use Agreement. Do not commit participant identifiers,
pickles, CSV datasets, NIfTI files, or notebook outputs containing restricted
records.

Local data can be stored outside the repository and selected with:

```bash
export FMRI_DATA_ROOT=/absolute/path/to/private/fmri-analysis-data
```

## Software

The main dependencies are Python 3.11+, NumPy, pandas, Nilearn, NiBabel,
NetworkX 3.5, SciPy, Matplotlib, joblib, and JupyterLab.

## Citation

If you reuse this workflow, cite the associated manuscript and acknowledge
PPMI according to its publication policy.
