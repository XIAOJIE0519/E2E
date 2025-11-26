# Generate and Plot SHAP Explanation Figures

Creates SHAP (SHapley Additive exPlanations) plots to explain feature
contributions by training a surrogate model on the original model's
scores.

## Usage

``` r
figure_shap(data, raw_data, target_type, file = NULL, model_type = "xgboost")
```

## Arguments

- data:

  A list containing `sample_score`, a data frame with sample IDs and
  `score`.

- raw_data:

  A data frame with original features. The first column must be the
  sample ID.

- target_type:

  String, the analysis type: "diagnosis" or "prognosis". This determines
  which columns in `raw_data` are treated as features.

- file:

  Optional. A string specifying the path to save the plot. If `NULL`
  (default), the plot object is returned.

- model_type:

  String, the surrogate model for SHAP calculation. "xgboost" (default)
  or "lasso".

## Value

A patchwork object combining SHAP summary and importance plots. If
`file` is provided, the plot is also saved.

## Examples

``` r
# \donttest{
# --- Example for a Diagnosis Model ---
set.seed(123)
train_dia_data <- data.frame(
  SampleID = paste0("S", 1:100),
  Label = sample(c(0, 1), 100, replace = TRUE),
  FeatureA = rnorm(100, 10, 2),
  FeatureB = runif(100, 0, 5)
)
model_results <- list(
  sample_score = data.frame(ID = paste0("S", 1:100), score = runif(100, 0, 1))
)

# Generate SHAP plot object
shap_plot <- figure_shap(
  data = model_results,
  raw_data = train_dia_data,
  target_type = "diagnosis",
  model_type = "xgboost"
)
#> Training 'xgboost' surrogate model and calculating SHAP values...
# To display the plot:
# print(shap_plot)
# }
```
