# Run Multiple Diagnostic Models

Trains and evaluates one or more registered diagnostic models on a given
dataset.

## Usage

``` r
models_dia(
  data,
  model = "all_dia",
  tune = FALSE,
  seed = 123,
  threshold_choices = "default",
  positive_label_value = 1,
  negative_label_value = 0,
  new_positive_label = "Positive",
  new_negative_label = "Negative"
)
```

## Arguments

- data:

  A data frame where the first column is the sample ID, the second is
  the outcome label, and subsequent columns are features.

- model:

  A character string or vector of character strings, specifying which
  models to run. Use "all_dia" to run all registered models.

- tune:

  Logical, whether to enable hyperparameter tuning for individual
  models.

- seed:

  An integer, for reproducibility of random processes.

- threshold_choices:

  A character string (e.g., "f1", "youden", "default") or a numeric
  value (0-1), or a named list/vector allowing different threshold
  strategies/values for each model.

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

A named list, where each element corresponds to a run model and contains
its trained `model_object`, `sample_score` data frame, and
`evaluation_metrics`.

## See also

[`initialize_modeling_system_dia`](https://xiaojie0519.github.io/E2E/reference/initialize_modeling_system_dia.md),
[`evaluate_model_dia`](https://xiaojie0519.github.io/E2E/reference/evaluate_model_dia.md)

## Examples

``` r
# \donttest{
# This example assumes your package includes a dataset named 'train_dia'.
# If not, you should create a toy data frame similar to the one below.
#
# train_dia <- data.frame(
#   ID = paste0("Patient", 1:100),
#   Disease_Status = sample(c(0, 1), 100, replace = TRUE),
#   FeatureA = rnorm(100),
#   FeatureB = runif(100)
# )

# Ensure the 'train_dia' dataset is available in the environment
# For example, if it is exported by your package:
# data(train_dia)

# Check if 'train_dia' exists, otherwise skip the example
if (exists("train_dia")) {
  # 1. Initialize the modeling system
  initialize_modeling_system_dia()

  # 2. Run selected models
  results <- models_dia(
    data = train_dia,
    model = c("rf", "lasso"), # Run only Random Forest and Lasso
    threshold_choices = list(rf = "f1", lasso = 0.6), # Different thresholds
    positive_label_value = 1,
    negative_label_value = 0,
    new_positive_label = "Case",
    new_negative_label = "Control",
    seed = 42
  )

  # 3. Print summaries
  for (model_name in names(results)) {
    print_model_summary_dia(model_name, results[[model_name]])
  }
}
#> Diagnostic modeling system already initialized
#> Running model: rf
#> Warning: ci.auc() of a ROC curve with AUC == 1 is always 1-1 and can be misleading.
#> Running model: lasso
#> 
#> --- rf Model (on Training Data) Metrics ---
#> Threshold Strategy: f1 (0.7540)
#> AUROC: 1.0000 (95% CI: 1.0000 - 1.0000)
#> AUPRC: 1.0000
#> Accuracy: 1.0000
#> F1: 1.0000
#> Precision: 1.0000
#> Recall: 1.0000
#> Specificity: 1.0000
#> --------------------------------------------------
#> 
#> --- lasso Model (on Training Data) Metrics ---
#> Threshold Strategy: numeric (0.6000)
#> AUROC: 0.9946 (95% CI: 0.9910 - 0.9983)
#> AUPRC: 0.9995
#> Accuracy: 0.9722
#> F1: 0.9847
#> Precision: 0.9822
#> Recall: 0.9872
#> Specificity: 0.8250
#> --------------------------------------------------
# }
```
