# Shipment ETA and delay propagation

A focused take-home system for predicting a shipment's final arrival and explaining how delay moves across stages.

The repository holds a synthetic set of 200 shipments and a model that predicts final delay once port of discharge is complete. The short submission PDF is still to be written from these notes.

## Delay, in one line

`final_delay_hours` is the actual end of inland delivery minus the scheduled end of inland delivery. Positive means late. Both times are UTC.

The rules, the stage list, the worked 26-hour example, and the checks on the 200 shipments are in [report/01_definitions_and_data.md](report/01_definitions_and_data.md).

## Run

Python 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m eta_delay
python -m eta_delay.evaluate
pytest
```

`python -m eta_delay` rewrites `data/shipments.csv` and `data/journeys.csv` from seed `20261111` and prints the sanity checks. The same seed produces the same rows.

`python -m eta_delay.evaluate` fits the decision-time model with five shipment-level folds and rewrites [report/02_model_results.md](report/02_model_results.md) and `data/oof_predictions.csv`.

`pytest` checks the hand-worked journey, the propagation identities, the leakage rule, and that the propagation model beats both baselines.

## Data

| File | Rows | Contents |
| --- | --- | --- |
| `data/shipments.csv` | 200 | One shipment: route, arrivals, final delay, congestion |
| `data/journeys.csv` | 1,200 | One stage of one shipment |
| `data/oof_predictions.csv` | 200 | Held-out delay predictions for every shipment |

`truth_*` columns are ground truth, kept so the mechanism can be audited. `observed_*` columns are the recorded milestones. Customs is blank there for some shipments. A model must not read future or unobserved truth timestamps as inputs.

## Layout

| Path | Role |
| --- | --- |
| `eta_delay/definitions.py` | Stages, buffers, routes, seed, delay sign |
| `eta_delay/generate.py` | Schedule, realized duration, propagation used to build the data |
| `eta_delay/features.py` | What is known at the end of port of discharge |
| `eta_delay/stage_model.py` | Ridge duration model for the remaining stages, and the flat comparison |
| `eta_delay/propagate.py` | Slack rule and the hours each stage still contributes |
| `eta_delay/evaluate.py` | Five-fold score and the explained shipment |
| `tests/` | Mechanism, leakage, and the baseline comparison |

Read `propagate.py` first if you want the explanation rule, then `features.py` for what the model is allowed to see.
