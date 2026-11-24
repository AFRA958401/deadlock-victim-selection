# Adaptive Four-Factor Victim Selection for Deadlock Recovery

Code and data for the paper **"An Adaptive Four-Factor Victim-Selection Policy for Operating-System
Deadlock Recovery: A Controlled Simulation Study"** (Afra Fatima, Rafia Rehan, Tuba Akbar Ali).

The simulator compares five victim-selection policies (Adaptive, Fixed Weight, Lowest Priority,
Minimum Cost, Minimum History), plus factor-ablation variants and an age-based baseline (Youngest First).

## Files

| File / folder | Purpose |
|---|---|
| `simulation.py` | Simulator and all victim-selection policies |
| `experiment_runner.py` | Main experiment (4,050 runs) and supplementary experiment (1,890 runs) |
| `run_age_baseline.py` | Supplementary age-based baseline, Youngest First (810 runs) |
| `analyze_results.py` | Tables 1-4: policy means, 32 paired comparisons (Wilcoxon, Holm, bootstrap CI), ablation means |
| `generate_paper_assets.py` | Table 5 and Figures 4-5 from the main run-level data |
| `age_compare.py` | Table 6: Adaptive versus Youngest First (Holm over 8 comparisons) |
| `main/`, `supplementary/`, `age_baseline/` | Run-level CSV files (one row per run) |
| `table3_statistical_results.csv`, `table4_supplementary_means.csv`, `table6_age_baseline.csv`, `main_policy_means.csv` | Generated tables |

## Requirements

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

## Reproduce the results

```bash
python experiment_runner.py --mode all --workers 4 --outdir .         # main + supplementary runs
python run_age_baseline.py                                              # age-based baseline
python analyze_results.py                                               # Tables 1-4
python age_compare.py                                                   # Table 6
python generate_paper_assets.py                                         # Table 5 + Figures 4-5
```

Every run is deterministic for a given configuration and seed. The analysis scripts read the CSV files in `main/`, `supplementary/`, and `age_baseline/`. The `--outdir .` option writes new main and supplementary CSVs into those same folders. The tables and figures can also be regenerated from the included CSVs without re-running the simulation. Bootstrap confidence intervals use random seed 20261123.

## Model settings

- Horizon 500 time units; arrivals in the first 80 time units
- 20, 50 or 100 processes; 3, 5 or 8 resource types; two instances per type
- Contention (probability of a second request): 0.2, 0.5, 0.8; seeds 0-29 (main), 0-9 (supplementary)
- Restart penalty 2, termination penalty 5, recovery overhead 1 per victim event
- A process is terminated after its fifth victimisation
- Victim candidates: only processes that lie on a cycle of the wait-for graph among the deadlocked processes
- Adaptive history weight: `clip(l0 + a*C + b*rho, 0.20, 0.60)` with default (l0, a, b) = (0.20, 0.20, 0.20)

## Provenance and AI assistance

`simulation.py` was written with the help of an AI assistant (Claude, Anthropic) from the model
specification in Section 3 of the paper, and the analysis scripts were written with the same help.
The reported results come from this code and these data files.

## Data integrity

SHA-256 of the run-level files: see `main/results_main.meta.json` and
`supplementary/results_supplementary.meta.json`.

## Implementation note

`simulation.py` was reconstructed from the model specification in the paper (see Provenance above).
The statistical analysis was regenerated from the run-level data; see `docs/RERUN_REPORT.txt` and
`docs/TECHNICAL_FIXES_AUDIT.txt` for details. Results may be sensitive to implementation details.

## Authors

Afra Fatima, Rafia Rehan, Tuba Akbar Ali - Department of Data Science, University of Management and Technology.
Supervisor: Mohsin Ghaffar Ghouri.

## License and citation

Code: MIT License (see `LICENSE`). Run-level data and generated tables: CC BY 4.0.
Citation metadata: `CITATION.cff`. Zenodo DOI: [TO BE FILLED AFTER RELEASE].
