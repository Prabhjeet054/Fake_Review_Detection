# Fake Review Detector

Three-class classifier for product and hotel reviews: **Genuine Human**, **Fake Human-Written**, or **AI-Generated**. Frozen DistilBERT `[CLS]` embeddings feed a small MLP head; TF-IDF baselines (SVM, trees, boosting) are trained for comparison. All settings live in `config.yaml` with seed **42**.

## Classes

| ID | Name | Source |
| --- | --- | --- |
| 0 | Genuine Human | Ott truthful hotel reviews + Salminen original (`OR`) product reviews |
| 1 | Fake Human-Written | Ott deceptive (`DECEPTIVE`) hotel reviews |
| 2 | AI-Generated | Salminen computer-generated (`CG`) product reviews |

## Datasets

Download both CSVs into `data/raw/` using the filenames from `config.yaml`:

| Dataset | Expected filename | Download |
| --- | --- | --- |
| Salminen Fake Reviews | `data/raw/fake_reviews_dataset.csv` | [OSF `tyue9`](https://osf.io/tyue9/) |
| Ott Deceptive Opinion Spam | `data/raw/deceptive-opinion.csv` | [Kaggle deceptive-opinion-spam-corpus](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus) |

## Setup

Requires **Python 3.11** (the version used for this case study). On macOS the `python` command is often missing; use `python3.11`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. If `python3.11` is not on your PATH, use the Python 3.11 executable you installed (`python` or `python3`).

## Run the pipeline

```bash
python run_all.py
python run_all.py --stages data,eda,baselines
python run_all.py --force
python run_all.py --stages embed,train,evaluate --force
```

Stages run in this order: `data` (build, clean, dedup, split, check), `eda`, `baselines`, `embed`, `train`, `evaluate`, `plots`, `analysis`, `explain`, `experiments`. `--stages` takes a comma-separated subset. A stage is skipped when all of its outputs already exist; `--force` re-runs those stages. The pipeline stops at the first failing stage with a non-zero exit code.

Optional GPU fine-tuning of the last two DistilBERT layers (requires CUDA or Apple MPS; on CPU the command exits with a clear error):

```bash
python -m src.experiments --finetune
```

Refresh the results table below after a pipeline run:

```bash
python scripts/update_readme.py
```

## Run the app

```bash
streamlit run app/streamlit_app.py
```

The app loads `head_best.pt` and the frozen encoder. Run `python run_all.py` through `train` first.

## Run tests

```bash
pytest -m "not slow"          # fixtures and unit tests; no real CSVs or DistilBERT
pytest                        # full suite, including real-data and encoder tests
pytest -m slow                # only slow tests
```

`@pytest.mark.slow` covers tests that need the real datasets, a DistilBERT download, or more than about 10 seconds.

## Project layout

```
config.yaml                 # seed, paths, model, training, experiments
run_all.py                  # one-command pipeline
scripts/update_readme.py    # fills the results table from metric files
src/                        # load, clean, split, baselines, DistilBERT, eval
app/streamlit_app.py        # interactive detector
tests/                      # pytest; conftest.py supplies a 30-row tmp config
data/raw/                   # download the two CSVs here
outputs/metrics/            # metrics.json, baselines.csv, experiments.csv
outputs/figures/            # curves, confusion matrix, explanations
outputs/models/             # head_best.pt, TF-IDF and baseline joblibs (gitignored)
```

## Results

Filled from `outputs/metrics/metrics.json` and `outputs/metrics/baselines.csv`.

<!-- RESULTS:START -->

| Model | Test accuracy | Test macro F1 | Cohen's kappa |
| --- | --- | --- | --- |
| DistilBERT + MLP | 0.8611 | 0.8518 | 0.7713 |
| SVM | 0.9000 | 0.8954 | — |
| Gradient Boosting | 0.8306 | 0.8226 | — |
| Random Forest | 0.8389 | 0.8137 | — |
| KNN | 0.7278 | 0.7357 | — |
| Decision Tree | 0.6542 | 0.6403 | — |
| AdaBoost | 0.6028 | 0.6158 | — |
| Majority class | 0.5000 | 0.2222 | — |

<!-- RESULTS:END -->
