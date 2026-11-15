# Fake Review Detector

Classify product reviews as Genuine Human, Fake Human-Written, or AI-Generated.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## Run the pipeline

```bash
python run_all.py
python run_all.py --stages data,eda,baselines
python run_all.py --force
python run_all.py --stages embed,train,evaluate --force
```

Stages run in this order: `data` (build, clean, dedup, split, check), `eda`,
`baselines`, `embed`, `train`, `evaluate`, `plots`, `analysis`, `explain`,
`experiments`. `--stages` takes a comma-separated subset. A stage is skipped
when all of its outputs already exist; `--force` re-runs those stages. The
pipeline stops at the first failing stage with a non-zero exit code and prints
data sizes, the best baseline test macro F1, DistilBERT test accuracy / macro F1
/ kappa, and per-stage runtime.

## Compare experiments

After preparing the splits and frozen embeddings, run:

```bash
python -m src.experiments
```

This trains the baseline, class-weighted, early-stopping, and mean-pooling
variants with seed 42. Results are written to
`outputs/metrics/experiments.csv` and `outputs/metrics/experiments.md`; mean
embeddings use separate `outputs/embeddings/mean_*` cache files. To include
top-two-layer fine-tuning, use `--finetune` on a CUDA or Apple GPU.
