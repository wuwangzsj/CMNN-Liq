# CMNN-Liq: core method implementation

Reference-state representation for pathwise monotonicity in CPT-based liquefaction probability models.

This is a **lightweight code release**, curated from the original research scripts. It shares the core method and implementation, not an end-to-end reproduction of the manuscript tables. No database records, fitted weights, archived predictions, or complete experiment runners are included. Original scientific expressions are retained; machine-specific paths and result-management code have been removed. See `SOURCE_INDEX.json` for provenance.

## Read in this order

| File | Purpose |
|---|---|
| `src/features.py` | Juang-based FS, reference FS and log coordinates |
| `src/cmnn.py`, `src/training.py` | CMNN layers and the main full-batch training kernel |
| `src/inference.py` | Inference from effective weights and PGA-path logit slope |
| `src/comparators.py` | Current-state A/N, M-noFS and constrained Logistic B0/B1 |
| `src/factorial.py`, `src/factorial_training.py` | Matched M0/R/C/A0 constraints and projection |
| `src/support_query.py` | Input support and per-network conditional PGA roots |
| `src/path_metrics.py` | Single-step decrease, peak drawdown and cumulative negative steps |
| `src/reference_mw.py` | Optional PGA-and-Mw reference-state extension |
| `src/evaluation.py` | Held-out prediction scoring |

The derivation and evaluation workflow are described in [METHOD.md](METHOD.md). Chinese notes are in [README_中文.md](README_中文.md).

## Minimal use

Install the imports in `requirements.txt` in your own Python environment. Run the non-training checks:

```bash
python check_core.py
```

To use the modules interactively, add `src` to Python's import path. Provide a pandas DataFrame with the fields in `data/input_schema.csv` (the file contains only a header). `pga` is dimensionless PGA/g, numerically the acceleration expressed in g; `depth` is in metres; `stress_ratio` is total/effective vertical stress; `qt1n` and `ic` are dimensionless. `y` is binary and `region` identifies a geographical group. Training and validation frames must be supplied separately.

For main-model training use `training.Repaired('CMNN', config, seed)`, with a candidate configuration returned by `training.candidates('CMNN', config)`. `fit(train, validation)` performs validation-based stopping; `fit(train, epochs=n)` uses a specified outer refit budget. **Do not use the legacy `CMNN.fit` directly for the manuscript training objective**: the wrapper initializes it with zero epochs, then applies the archived explicit-L2 objective. No training is run by importing a module or by `check_core.py`.

`factorial_training.train_fixed` retains the matched-training kernel. The internal geometry labels `G1` and `G3` correspond to M0 and A0. M is the inner-selected main network; M0 is a distinct fixed-configuration comparison. Do not interchange their results.

`reference_mw.activate_adapter()` switches the layer and trainer classes in its process. Run this extension in a separate process from main-model training. It applies only when Mw enters FS through the stated MSF term.

## Data and availability

The source case-history database is LIQ/7/2833, version 3: https://doi.org/10.17632/rmjnghdrbv.3 . Obtain the original data from its repository and follow its terms. The input schema is not the original workbook schema or a preprocessing recipe. This code-only release does not claim to reproduce the published scores without the original splits, processed cases and archived model artifacts.

The supplied modules implement the paper's engineering formulation and CMNN construction. Relevant sources include Runje and Shankaranarayana (2023), *Constrained Monotonic Neural Networks*, and the Juang/Liq-HBM FS implementation discussed in the manuscript. This release is not the official code of those authors.

## License

No open-source license has been selected for this prepared release. The authors should set one before granting reuse rights. Third-party data retain their own terms. Repository URL, release DOI and final manuscript citation should be added after publication/upload; none is invented here.
