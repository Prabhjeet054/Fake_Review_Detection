# Fake Review Detector

University case study: **Detecting Fake or AI-Generated Product Reviews Using Deep Learning**.

The project classifies a product review into one of three classes:

| ID | Class |
| --- | --- |
| 0 | Genuine Human Review |
| 1 | Fake Human-Written Review |
| 2 | AI-Generated Review |

Every path, seed, and hyperparameter is read from `config.yaml`. The global seed is **42**.

## Method

1. **Data.** Combine the [Salminen Fake Reviews Dataset](https://osf.io/tyue9/) (`OR` → 0, `CG` → 2) with the [Ott Deceptive Opinion Spam Corpus](http://myleott.com/op_spam/) (`truthful` → 0, `deceptive` → 1).
2. **Preprocessing.** Strip HTML, drop special symbols (keep `. , ! ? '`), expand contractions, lowercase, collapse whitespace.
3. **Leakage control.** Near-duplicate grouping (Jaccard / MinHash), then a stratified, group-aware 70 / 15 / 15 train / val / test split.
4. **Baselines.** TF-IDF (1–2 grams) with KNN, Decision Tree, Random Forest, SVM, AdaBoost, and Gradient Boosting.
5. **Main model.** Frozen `distilbert-base-uncased` (max length 128) → `[CLS]` embedding (768-d) → `Linear(768, 256)` → ReLU → Dropout(0.3) → `Linear(256, 3)`. Embeddings are computed once and cached. The head is trained with Adam (`lr=1e-4`), batch size 32, 50 epochs, and cross-entropy.
6. **Evaluation.** Accuracy, per-class precision / recall / F1, macro and weighted averages, Cohen's kappa, confusion matrices, grouped binary metrics, and error analysis.

## Setup

Python 3.10–3.12 is recommended (3.11 is a good default). System Python 3.14 may not have wheels for PyTorch yet.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `PYTHONHASHSEED=42` when you run the pipeline so hash-based operations stay reproducible.

## Data layout

Download the datasets and place them here:

```
data/raw/salminen/fake_reviews_dataset.csv
data/raw/ott/          # extracted Ott corpus (nested truthful/deceptive folders)
```

Salminen is the OSF file `fake reviews dataset.csv` (rename to `fake_reviews_dataset.csv`). Ott is a tree of `.txt` files whose parent folders contain `truthful` or `deceptive`.

## Run the pipeline

```bash
PYTHONHASHSEED=42 python run_all.py
```

Useful variants:

```bash
# Data preparation only
PYTHONHASHSEED=42 python run_all.py --steps load clean dedup split eda check

# Skip the slow classical models
PYTHONHASHSEED=42 python run_all.py --skip baselines

# Resume from embedding cache
PYTHONHASHSEED=42 python run_all.py --from-step embed
```

Each module is also runnable on its own:

```bash
PYTHONHASHSEED=42 python -m src.load_data
PYTHONHASHSEED=42 python -m src.clean
PYTHONHASHSEED=42 python -m src.dedup
PYTHONHASHSEED=42 python -m src.split
PYTHONHASHSEED=42 python -m src.eda
PYTHONHASHSEED=42 python -m src.check_data
PYTHONHASHSEED=42 python -m src.features
PYTHONHASHSEED=42 python -m src.baselines
PYTHONHASHSEED=42 python -m src.tokenization
PYTHONHASHSEED=42 python -m src.embed
PYTHONHASHSEED=42 python -m src.model
PYTHONHASHSEED=42 python -m src.train
PYTHONHASHSEED=42 python -m src.evaluate
PYTHONHASHSEED=42 python -m src.analysis
PYTHONHASHSEED=42 python -m src.predict --text "This blender is great."
PYTHONHASHSEED=42 python -m src.explain --text "This product is amazing. Best purchase ever."
PYTHONHASHSEED=42 python -m src.experiments
PYTHONHASHSEED=42 python -m src.results_summary
```

## Tests

```bash
PYTHONHASHSEED=42 pytest
```

Tests use tiny fixtures under `tests/fixtures/` and do not download DistilBERT.

## Streamlit demo

After training (`outputs/models/review_classifier.pt` must exist):

```bash
streamlit run app/streamlit_app.py
```

Paste a review to see the predicted class, probabilities, and an optional token-occlusion explanation.

## Project layout

```
config.yaml
requirements.txt
run_all.py
src/          # pipeline modules (never named tokenize.py)
app/streamlit_app.py
data/raw | interim | processed/splits
outputs/figures | metrics | models | embeddings | logs
tests/        # pytest, with fixtures in tests/fixtures
```
