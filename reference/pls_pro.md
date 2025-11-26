# Train Partial Least Squares Cox (PLS-Cox)

Fits a Cox model using Partial Least Squares reduction for
high-dimensional data.

## Usage

``` r
pls_pro(X, y_surv, tune = FALSE)
```

## Arguments

- X:

  A data frame of predictors.

- y_surv:

  A `Surv` object containing time and status.

- tune:

  Logical. If TRUE, performs internal tuning (currently handled by
  cv.glmnet automatically).

## Value

An object of class `survival_plsRcox` and `pro_model`.
