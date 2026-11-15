# Report numbers

Every number below is labelled with the file it was read from.

## Dataset and class description

*Source: `outputs/metrics/data_card.md`*

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

## Test-set size and class counts

*Source: `outputs/metrics/split_summary.json`*

| split | n_rows | class_0 | class_1 | class_2 |
| --- | --- | --- | --- | --- |
| train | 3356 | 1676 | 560 | 1120 |
| val | 720 | 360 | 120 | 240 |
| test | 720 | 360 | 120 | 240 |

- Exact test-set size: **720**
- Test class counts: **0=360, 1=120, 2=240**

## Model configuration

*Source: `config.yaml`*

| setting | value |
| --- | --- |
| seed | 42 |
| model.name | distilbert-base-uncased |
| model.max_length | 128 |
| model.freeze_encoder | True |
| model.hidden_units | 256 |
| model.dropout | 0.3000 |
| model.num_classes | 3 |
| training.learning_rate | 0.0001 |
| training.batch_size | 32 |
| training.epochs | 50 |
| training.embed_batch_size | 64 |

## Parameter counts

*Source: `outputs/metrics/model_summary.txt`*

| count | parameters |
| --- | --- |
| total | 66560515 |
| trainable | 197635 |
| non_trainable | 66362880 |

## Training history

*Source: `outputs/metrics/history.csv`*

- Best epoch (lowest val_loss): **46**

| note | epoch | train_loss | train_acc | val_loss | val_acc |
| --- | --- | --- | --- | --- | --- |
| epoch 1 | 1 | 0.7805 | 0.6666 | 0.7813 | 0.6514 |
| epoch 10 | 10 | 0.3674 | 0.8510 | 0.3877 | 0.8486 |
| epoch 20 | 20 | 0.3089 | 0.8707 | 0.3495 | 0.8625 |
| epoch 30 | 30 | 0.2688 | 0.8874 | 0.3297 | 0.8667 |
| epoch 50 | 50 | 0.2180 | 0.9133 | 0.3164 | 0.8639 |
| best (epoch 46, min val_loss) | 46 | 0.2236 | 0.9091 | 0.3116 | 0.8667 |

## Test results

*Source: `outputs/metrics/metrics.json`*

| n_test | n_correct | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 | cohen_kappa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 720 | 620 | 0.8611 | 0.8586 | 0.8458 | 0.8518 | 0.8610 | 0.8611 | 0.8608 | 0.7713 |

### Per-class

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| Genuine Human | 0.8537 | 0.8750 | 0.8642 | 360 |
| Fake Human-Written | 0.8393 | 0.7833 | 0.8103 | 120 |
| AI-Generated | 0.8828 | 0.8792 | 0.8810 | 240 |

### Confusion matrix

| true | Genuine Human | Fake Human-Written | AI-Generated |
| --- | --- | --- | --- |
| Genuine Human | 315 | 18 | 27 |
| Fake Human-Written | 25 | 94 | 1 |
| AI-Generated | 29 | 0 | 211 |

## Grouped binary metrics

*Source: `outputs/metrics/binary_metrics.json`*

| task | accuracy | precision | recall |
| --- | --- | --- | --- |
| genuine_vs_deceptive | 0.8625 | 0.8718 | 0.8500 |
| human_vs_ai_generated | 0.9208 | 0.8828 | 0.8792 |

## Error breakdown

*Source: `outputs/metrics/error_breakdown.csv`*

| true_class | predicted_class | count | share_of_errors |
| --- | --- | --- | --- |
| 2 | 0.0000 | 29 | 0.2900 |
| 0.0000 | 2 | 27 | 0.2700 |
| 1 | 0.0000 | 25 | 0.2500 |
| 0.0000 | 1 | 18 | 0.1800 |
| 1 | 2 | 1 | 0.0100 |
| 2 | 1 | 0.0000 | 0.0000 |

## Baselines (val and test)

*Source: `outputs/metrics/baselines.csv`*

### val

| model | model_key | split | train_seconds | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority class | majority_class | val | 0.0002 | 0.5000 | 0.1667 | 0.3333 | 0.2222 | 0.3333 |
| KNN | knn | val | 0.0006 | 0.6958 | 0.7148 | 0.7583 | 0.7066 | 0.6817 |
| Decision Tree | decision_tree | val | 0.4006 | 0.6736 | 0.6588 | 0.6593 | 0.6582 | 0.6723 |
| Random Forest | random_forest | val | 0.4478 | 0.8083 | 0.8284 | 0.7657 | 0.7895 | 0.8062 |
| SVM | svm | val | 0.0389 | 0.8569 | 0.8599 | 0.8454 | 0.8522 | 0.8568 |
| AdaBoost | adaboost | val | 73.0929 | 0.5986 | 0.5945 | 0.6435 | 0.6107 | 0.5953 |
| Gradient Boosting | gradient_boosting | val | 568.3478 | 0.8153 | 0.8108 | 0.8083 | 0.8095 | 0.8152 |

### test

| model | model_key | split | train_seconds | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority class | majority_class | test | 0.0002 | 0.5000 | 0.1667 | 0.3333 | 0.2222 | 0.3333 |
| KNN | knn | test | 0.0006 | 0.7278 | 0.7411 | 0.7852 | 0.7357 | 0.7169 |
| Decision Tree | decision_tree | test | 0.4006 | 0.6542 | 0.6567 | 0.6292 | 0.6403 | 0.6529 |
| Random Forest | random_forest | test | 0.4478 | 0.8389 | 0.8650 | 0.7861 | 0.8137 | 0.8355 |
| SVM | svm | test | 0.0389 | 0.9000 | 0.9033 | 0.8884 | 0.8954 | 0.8998 |
| AdaBoost | adaboost | test | 73.0929 | 0.6028 | 0.5980 | 0.6542 | 0.6158 | 0.5987 |
| Gradient Boosting | gradient_boosting | test | 568.3478 | 0.8306 | 0.8387 | 0.8097 | 0.8226 | 0.8301 |

## Experiments

*Source: `outputs/metrics/experiments.csv`*

| variant | val_accuracy | val_macro_f1 | val_genuine_precision | val_genuine_recall | test_accuracy | test_macro_f1 | test_genuine_precision | test_genuine_recall | epochs_trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8667 | 0.8552 | 0.8750 | 0.8556 | 0.8611 | 0.8518 | 0.8537 | 0.8750 | 50 |
| class_weighted | 0.8583 | 0.8503 | 0.9082 | 0.7972 | 0.8667 | 0.8593 | 0.8982 | 0.8333 | 50 |
| early_stopping | 0.8667 | 0.8552 | 0.8750 | 0.8556 | 0.8611 | 0.8518 | 0.8537 | 0.8750 | 50 |
| mean_pooling | 0.8653 | 0.8558 | 0.8623 | 0.8694 | 0.8750 | 0.8693 | 0.8668 | 0.8861 | 50 |

## Figure files

*Source: `outputs/figures`*

- `outputs/figures/accuracy_curve.png`
- `outputs/figures/baselines_test.png`
- `outputs/figures/baselines_val.png`
- `outputs/figures/class_distribution.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/confusion_matrix_normalized.png`
- `outputs/figures/explanations/id_0_true_1_pred_1.html`
- `outputs/figures/explanations/id_135_true_2_pred_0.html`
- `outputs/figures/explanations/id_142_true_1_pred_0.html`
- `outputs/figures/explanations/id_260_true_0_pred_2.html`
- `outputs/figures/explanations/id_30_true_2_pred_2.html`
- `outputs/figures/explanations/id_38_true_0_pred_0.html`
- `outputs/figures/explanations/id_40_true_0_pred_0.html`
- `outputs/figures/explanations/id_47_true_2_pred_2.html`
- `outputs/figures/explanations/id_82_true_1_pred_1.html`
- `outputs/figures/length_distribution.png`
- `outputs/figures/loss_curve.png`
- `outputs/figures/per_class_metrics.png`
- `outputs/figures/token_lengths.png`

## Consistency checks

*Source: `computed from the files above`*

- PASS: metrics.json vs confusion matrix
- PASS: history.csv last row vs plotted curves
- PASS: head_best.pt config vs config.yaml
- PASS: head parameter count (197,635)
- PASS: two seed-42 test accuracies
