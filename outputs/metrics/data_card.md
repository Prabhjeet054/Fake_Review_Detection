# Data card

## Dataset description

| Field | Value |
| --- | --- |
| Sources | Salminen Fake Reviews Dataset (`fake_reviews_dataset.csv`); Ott Deceptive Opinion Spam Corpus (`deceptive-opinion.csv`) |
| Input | `data/processed/dataset.csv` |
| Number of classes | 3 |
| Split ratios | train 0.70 / val 0.15 / test 0.15 |
| Total rows | 4796 |
| Rows per split | train 3356; val 720; test 720 |
| Class counts per split | train: 0=1676, 1=560, 2=1120; val: 0=360, 1=120, 2=240; test: 0=360, 1=120, 2=240 |
| Class percentages in the test split | 0 Genuine Human 50.0%; 1 Fake Human-Written 16.7%; 2 AI-Generated 33.3% |

## Class description

| Label | Class name | Source | Domain | Count in test |
| --- | --- | --- | --- | --- |
| 0 | Genuine Human | ott, salminen | hotel, product | 360 |
| 1 | Fake Human-Written | ott | hotel | 120 |
| 2 | AI-Generated | salminen | product | 240 |
