# Comprehensive Diagnostic Modeling Pipeline

Executes a complete diagnostic modeling workflow including single
models, bagging, stacking, and voting ensembles across training and
multiple test datasets. Returns structured results with AUROC values for
visualization.

## Usage

``` r
int_dia(
  ...,
  model_names = NULL,
  tune = TRUE,
  n_estimators = 10,
  seed = 123,
  positive_label_value = 1,
  negative_label_value = 0,
  new_positive_label = "Positive",
  new_negative_label = "Negative"
)
```

## Arguments

- ...:

  Data frames for analysis. The first is the training dataset; all
  subsequent arguments are test datasets.

- model_names:

  Character vector specifying which models to use. If NULL (default),
  uses all registered models.

- tune:

  Logical, enable hyperparameter tuning. Default TRUE.

- n_estimators:

  Integer, number of bootstrap samples for bagging. Default 10.

- seed:

  Integer for reproducibility. Default 123.

- positive_label_value:

  Value representing positive class. Default 1.

- negative_label_value:

  Value representing negative class. Default 0.

- new_positive_label:

  Factor level name for positive class. Default "Positive".

- new_negative_label:

  Factor level name for negative class. Default "Negative".

## Value

A list containing all_results, auroc_matrix, model_categories,
dataset_names.
