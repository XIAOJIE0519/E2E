# Comprehensive Prognostic Modeling Pipeline

Executes a complete prognostic (survival) modeling workflow including
single models, bagging, and stacking ensembles. Returns C-index and
time-dependent AUROC metrics.

## Usage

``` r
int_pro(
  ...,
  model_names = NULL,
  tune = TRUE,
  n_estimators = 10,
  seed = 123,
  time_unit = "day",
  years_to_evaluate = c(1, 3, 5)
)
```

## Arguments

- ...:

  Data frames for survival analysis. First = training; others = test
  sets. Format: first column = ID, second = outcome (0/1), third = time,
  remaining = features.

- model_names:

  Character vector specifying which models to use. If NULL (default),
  uses all registered prognostic models.

- tune:

  Logical, enable tuning. Default TRUE.

- n_estimators:

  Integer, bagging iterations. Default 10.

- seed:

  Integer for reproducibility. Default 123.

- time_unit:

  Time unit in data: "day", "month", or "year". Default "day".

- years_to_evaluate:

  Numeric vector of years for time-dependent AUROC. Default c(1,3,5).

## Value

A list with:

- `all_results`: All model outputs

- `cindex_matrix`: C-index values (models × datasets)

- `avg_auroc_matrix`: Average time-dependent AUROC (models × datasets)

- `model_categories`: Model category labels

- `dataset_names`: Dataset identifiers

## Examples

``` r
if (FALSE) { # \dontrun{
prognosis_results <- int_pro(train_pro, test_pro1, test_pro2)
} # }
```
