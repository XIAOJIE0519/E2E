# Train Random Survival Forest (RSF)

Fits a Random Survival Forest using the log-rank splitting rule.
Includes capabilities for hyperparameter tuning via grid search over
`ntree`, `nodesize`, and `mtry`.

## Usage

``` r
rsf_pro(X, y_surv, tune = FALSE, tune_params = NULL)
```

## Arguments

- X:

  A data frame of predictors.

- y_surv:

  A `Surv` object containing time and status.

- tune:

  Logical. If TRUE, performs grid search for optimal hyperparameters
  based on C-index.

- tune_params:

  Optional data frame containing the grid for tuning.

## Value

An object of class `survival_rsf` and `pro_model`.
