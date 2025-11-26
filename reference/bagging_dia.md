# Train a Bagging Diagnostic Model

Implements a Bagging (Bootstrap Aggregating) ensemble for diagnostic
models. It trains multiple base models on bootstrapped samples of the
training data and aggregates their predictions by averaging
probabilities.

## Usage

``` r
bagging_dia(
  data,
  base_model_name,
  n_estimators = 50,
  subset_fraction = 0.632,
  tune_base_model = FALSE,
  threshold_choices = "default",
  positive_label_value = 1,
  negative_label_value = 0,
  new_positive_label = "Positive",
  new_negative_label = "Negative",
  seed = 456
)
```

## Arguments

- data:

  A data frame where the first column is the sample ID, the second is
  the outcome label, and subsequent columns are features.

- base_model_name:

  A character string, the name of the base diagnostic model to use
  (e.g., "rf", "lasso"). This model must be registered.

- n_estimators:

  An integer, the number of base models to train.

- subset_fraction:

  A numeric value between 0 and 1, the fraction of samples to bootstrap
  for each base model.

- tune_base_model:

  Logical, whether to enable tuning for each base model.

- threshold_choices:

  A character string (e.g., "f1", "youden", "default") or a numeric
  value (0-1) for determining the evaluation threshold for the ensemble.

- positive_label_value:

  A numeric or character value in the raw data representing the positive
  class.

- negative_label_value:

  A numeric or character value in the raw data representing the negative
  class.

- new_positive_label:

  A character string, the desired factor level name for the positive
  class (e.g., "Positive").

- new_negative_label:

  A character string, the desired factor level name for the negative
  class (e.g., "Negative").

- seed:

  An integer, for reproducibility.

## Value

A list containing the `model_object`, `sample_score`, and
`evaluation_metrics`.

## See also

[`initialize_modeling_system_dia`](https://xiaojie0519.github.io/E2E/reference/initialize_modeling_system_dia.md),
[`evaluate_model_dia`](https://xiaojie0519.github.io/E2E/reference/evaluate_model_dia.md)

## Examples

``` r
# \donttest{
# This example assumes your package includes a dataset named 'train_dia'.
# If not, create a toy data frame first.
if (exists("train_dia")) {
  initialize_modeling_system_dia()

  bagging_rf_results <- bagging_dia(
    data = train_dia,
    base_model_name = "rf",
    n_estimators = 5, # Reduced for a quick example
    threshold_choices = "youden",
    positive_label_value = 1,
    negative_label_value = 0,
    new_positive_label = "Case",
    new_negative_label = "Control"
  )
  print_model_summary_dia("Bagging (RF)", bagging_rf_results)
}
#> Diagnostic modeling system initialized and default models registered.
#> Running Bagging model: Bagging_dia (base: rf)
#> Loading required package: ggplot2
#> Loading required package: lattice
#> 
#> --- Bagging (RF) Model (on Training Data) Metrics ---
#> Ensemble Type: Bagging (Base: rf, Estimators: 5)
#> Threshold Strategy: youden (0.7732)
#> AUROC: 0.9999 (95% CI: 0.9997 - 1.0000)
#> AUPRC: 1.0000
#> Accuracy: 0.9942
#> F1: 0.9968
#> Precision: 1.0000
#> Recall: 0.9936
#> Specificity: 1.0000
#> --------------------------------------------------
# }
```
