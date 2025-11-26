# Train Gradient Boosting Machine (GBM) for Survival

Fits a stochastic gradient boosting model using the Cox Partial
Likelihood distribution. Supports random search for hyperparameter
optimization.

## Usage

``` r
gbm_pro(X, y_surv, tune = FALSE, cv.folds = 5, max_tune_iter = 10)
```

## Arguments

- X:

  A data frame of predictors.

- y_surv:

  A `Surv` object.

- tune:

  Logical. If TRUE, performs random search.

- cv.folds:

  Integer. Number of cross-validation folds.

- max_tune_iter:

  Integer. Maximum iterations for random search.

## Value

An object of class `survival_gbm` and `pro_model`.
