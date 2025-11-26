# Train an EasyEnsemble Model for Imbalanced Classification

Implements the EasyEnsemble algorithm. It trains multiple base models on
balanced subsets of the data (by undersampling the majority class) and
aggregates their predictions.

## Usage

``` r
imbalance_dia(
  data,
  base_model_name = "xb",
  n_estimators = 10,
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
  (e.g., "xb", "rf"). This model must be registered.

- n_estimators:

  An integer, the number of base models to train (number of subsets).

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
# 1. Initialize the modeling system
initialize_modeling_system_dia()
#> Diagnostic modeling system already initialized

# 2. Create an imbalanced toy dataset
set.seed(42)
n_obs <- 100
n_minority <- 10
data_imbalanced_toy <- data.frame(
  ID = paste0("Sample", 1:n_obs),
  Status = c(rep(1, n_minority), rep(0, n_obs - n_minority)),
  Feat1 = rnorm(n_obs),
  Feat2 = runif(n_obs)
)

# 3. Run the EasyEnsemble algorithm
# n_estimators is reduced for a quick example
easyensemble_results <- imbalance_dia(
  data = data_imbalanced_toy,
  base_model_name = "xb",
  n_estimators = 3,
  threshold_choices = "f1"
)
#> Running Imbalance model: EasyEnsemble_dia (base: xb)
print_model_summary_dia("EasyEnsemble (XGBoost)", easyensemble_results)
#> 
#> --- EasyEnsemble (XGBoost) Model (on Training Data) Metrics ---
#> Ensemble Type: EasyEnsemble (Base: xb, Estimators: 3)
#> Threshold Strategy: f1 (0.7441)
#> AUROC: 0.9233 (95% CI: 0.8662 - 0.9805)
#> AUPRC: 0.4266
#> Accuracy: 0.9200
#> F1: 0.6364
#> Precision: 0.5833
#> Recall: 0.7000
#> Specificity: 0.9444
#> --------------------------------------------------
# }
```
