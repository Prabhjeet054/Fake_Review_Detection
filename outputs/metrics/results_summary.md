# Fake Review Detector Results

Best experiment by **val_macro_f1**: `mean_pooling` (0.8558).

## Model comparison

| variant | val_accuracy | val_macro_f1 | val_genuine_precision | val_genuine_recall | test_accuracy | test_macro_f1 | test_genuine_precision | test_genuine_recall | epochs_trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8667 | 0.8552 | 0.8750 | 0.8556 | 0.8611 | 0.8518 | 0.8537 | 0.8750 | 50 |
| class_weighted | 0.8583 | 0.8503 | 0.9082 | 0.7972 | 0.8667 | 0.8593 | 0.8982 | 0.8333 | 50 |
| early_stopping | 0.8667 | 0.8552 | 0.8750 | 0.8556 | 0.8611 | 0.8518 | 0.8537 | 0.8750 | 50 |
| mean_pooling | 0.8653 | 0.8558 | 0.8623 | 0.8694 | 0.8750 | 0.8693 | 0.8668 | 0.8861 | 50 |
