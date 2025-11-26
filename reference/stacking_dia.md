# Train a Stacking Diagnostic Model

Implements a Stacking ensemble. It trains multiple base models, then
uses their predictions as features to train a meta-model.

## Usage

``` r
stacking_dia(
  results_all_models,
  data,
  meta_model_name,
  top = 5,
  tune_meta = FALSE,
  threshold_choices = "f1",
  seed = 789,
  positive_label_value = 1,
  negative_label_value = 0,
  new_positive_label = "Positive",
  new_negative_label = "Negative"
)
```

## Arguments

- results_all_models:

  A list of results from
  [`models_dia()`](https://xiaojie0519.github.io/E2E/reference/models_dia.md),
  containing trained base model objects and their evaluation metrics.

- data:

  A data frame where the first column is the sample ID, the second is
  the outcome label, and subsequent columns are features. Used for
  training the meta-model.

- meta_model_name:

  A character string, the name of the meta-model to use (e.g., "lasso",
  "gbm"). This model must be registered.

- top:

  An integer, the number of top-performing base models (ranked by AUROC)
  to select for the stacking ensemble.

- tune_meta:

  Logical, whether to enable tuning for the meta-model.

- threshold_choices:

  A character string (e.g., "f1", "youden", "default") or a numeric
  value (0-1) for determining the evaluation threshold for the ensemble.

- seed:

  An integer, for reproducibility.

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

## Value

A list containing the `model_object`, `sample_score`, and
`evaluation_metrics`.

## See also

[`models_dia`](https://xiaojie0519.github.io/E2E/reference/models_dia.md),
[`evaluate_model_dia`](https://xiaojie0519.github.io/E2E/reference/evaluate_model_dia.md)

## Examples

``` r
# \donttest{
# 1. Initialize the modeling system
initialize_modeling_system_dia()
#> Diagnostic modeling system already initialized

# 2. Create a toy dataset for demonstration
set.seed(42)
data_toy <- data.frame(
  ID = paste0("Sample", 1:60),
  Status = sample(c(0, 1), 60, replace = TRUE),
  Feat1 = rnorm(60),
  Feat2 = runif(60)
)

# 3. Generate mock base model results (as if from models_dia)
# In a real scenario, you would run models_dia() on your full dataset
base_model_results <- models_dia(
  data = data_toy,
  model = c("rf", "lasso"),
  seed = 123
)
#> Running model: rf
#> Warning: ci.auc() of a ROC curve with AUC == 1 is always 1-1 and can be misleading.
#> Running model: lasso

# 4. Run the stacking ensemble
stacking_results <- stacking_dia(
  results_all_models = base_model_results,
  data = data_toy,
  meta_model_name = "gbm",
  top = 2,
  threshold_choices = "f1"
)
#> Running Stacking model: Stacking_dia (meta: gbm)
#> Warning: ci.auc() of a ROC curve with AUC == 1 is always 1-1 and can be misleading.
print_model_summary_dia("Stacking (GBM)", stacking_results)
#> 
#> --- Stacking (GBM) Model (on Training Data) Metrics ---
#> Ensemble Type: Stacking (Meta: gbm, Base models used: rf, lasso)
#> Threshold Strategy: f1 (1.0000)
#> AUROC: 1.0000 (95% CI: 1.0000 - 1.0000)
#> AUPRC: 1.0000
#> Accuracy: 1.0000
#> F1: 1.0000
#> Precision: 1.0000
#> Recall: 1.0000
#> Specificity: 1.0000
#> --------------------------------------------------
# }
```
