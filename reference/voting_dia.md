# Train a Voting Ensemble Diagnostic Model

Implements a Voting ensemble, combining predictions from multiple base
models through soft or hard voting.

## Usage

``` r
voting_dia(
  results_all_models,
  data,
  type = c("soft", "hard"),
  weight_metric = "AUROC",
  top = 5,
  seed = 789,
  threshold_choices = "f1",
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
  evaluation.

- type:

  A character string, "soft" for weighted average of probabilities or
  "hard" for majority class voting.

- weight_metric:

  A character string, the metric to use for weighting base models in
  soft voting (e.g., "AUROC", "F1"). Ignored for hard voting.

- top:

  An integer, the number of top-performing base models (ranked by
  `weight_metric`) to include in the ensemble.

- seed:

  An integer, for reproducibility.

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
base_model_results <- models_dia(
  data = data_toy,
  model = c("rf", "lasso"),
  seed = 123
)
#> Running model: rf
#> Warning: ci.auc() of a ROC curve with AUC == 1 is always 1-1 and can be misleading.
#> Running model: lasso

# 4. Run the soft voting ensemble
soft_voting_results <- voting_dia(
  results_all_models = base_model_results,
  data = data_toy,
  type = "soft",
  weight_metric = "AUROC",
  top = 2,
  threshold_choices = "f1"
)
#> Running Voting model: Voting_dia (type: soft)
#> Warning: ci.auc() of a ROC curve with AUC == 1 is always 1-1 and can be misleading.
print_model_summary_dia("Soft Voting", soft_voting_results)
#> 
#> --- Soft Voting Model (on Training Data) Metrics ---
#> Ensemble Type: Voting (Type: soft, Weight Metric: AUROC, Base models used: rf, lasso)
#> Threshold Strategy: f1 (0.5622)
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
