# Print Diagnostic Model Summary

Prints a formatted summary of the evaluation metrics for a diagnostic
model, either from training data or new data evaluation.

## Usage

``` r
print_model_summary_dia(model_name, results_list, on_new_data = FALSE)
```

## Arguments

- model_name:

  A character string, the name of the model (e.g., "rf", "Bagging
  (RF)").

- results_list:

  A list containing model evaluation results, typically an element from
  the output of
  [`models_dia()`](https://xiaojie0519.github.io/E2E/reference/models_dia.md)
  or the result of
  [`bagging_dia()`](https://xiaojie0519.github.io/E2E/reference/bagging_dia.md),
  [`stacking_dia()`](https://xiaojie0519.github.io/E2E/reference/stacking_dia.md),
  [`voting_dia()`](https://xiaojie0519.github.io/E2E/reference/voting_dia.md),
  or
  [`imbalance_dia()`](https://xiaojie0519.github.io/E2E/reference/imbalance_dia.md).
  It must contain `evaluation_metrics` and `model_object` (if
  applicable).

- on_new_data:

  Logical, indicating whether the results are from applying the model to
  new, unseen data (`TRUE`) or from the training/internal validation
  data (`FALSE`).

## Value

NULL. Prints the summary to the console.

## Examples

``` r
# Example for a successfully evaluated model
successful_results <- list(
  evaluation_metrics = list(
    Threshold_Strategy = "f1",
    `_Threshold` = 0.45,
    AUROC = 0.85, AUROC_95CI_Lower = 0.75, AUROC_95CI_Upper = 0.95,
    AUPRC = 0.80, Accuracy = 0.82, F1 = 0.78,
    Precision = 0.79, Recall = 0.77, Specificity = 0.85
  )
)
print_model_summary_dia("MyAwesomeModel", successful_results)
#> 
#> --- MyAwesomeModel Model (on Training Data) Metrics ---
#> 
#> AUROC: 0.8500 (95% CI: 0.7500 - 0.9500)
#> AUPRC: 0.8000
#> Accuracy: 0.8200
#> F1: 0.7800
#> Precision: 0.7900
#> Recall: 0.7700
#> Specificity: 0.8500
#> --------------------------------------------------

# Example for a failed model
failed_results <- list(evaluation_metrics = list(error = "Training failed"))
print_model_summary_dia("MyFailedModel", failed_results)
#> Model: MyFailedModel | Status: Failed (Training failed)
```
