# Train Ridge Cox Model

Fits a Cox model with Ridge (L2) regularization.

## Usage

``` r
ridge_pro(X, y_surv, tune = FALSE)
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

An object of class `survival_glmnet` and `pro_model`.
