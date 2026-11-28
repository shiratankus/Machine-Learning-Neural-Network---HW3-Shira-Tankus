# Machine-Learning-Neural-Network---HW3-Shira-Tankus

## Task 1 — CIFAR-10 classification

Install the dependencies and run both models on CPU:

```bash
python -m pip install torch torchvision numpy matplotlib scikit-learn
python task1.py --device cpu
```

The script downloads CIFAR-10 if needed. By default, it evaluates four
configurations per model using three stratified folds, with 15 epochs per fit
and seed 42. It retrains each selected configuration on the full training split
and evaluates it on the official test split. Use `--model mlp` or `--model cnn`
to run one architecture. New runs save their results in `outputs/task1/` by
default. Use `--out-dir` to choose another destination.

The saved 15-epoch CPU results used in the report are available below.

| Model | Test accuracy | Summary | Hyperparameter search | First-layer weights |
| --- | ---: | --- | --- | --- |
| MLP | 57.98% | [JSON](outputs/mlp_full/mlp_summary.json) | [CSV](outputs/mlp_full/mlp_hyperparam_search.csv) | [PNG](outputs/mlp_full/mlp_first_layer_weights.png) |
| CNN | 74.67% | [JSON](outputs/cnn_full/cnn_summary.json) | [CSV](outputs/cnn_full/cnn_hyperparam_search.csv) | [PNG](outputs/cnn_full/cnn_first_layer_kernels.png) |
