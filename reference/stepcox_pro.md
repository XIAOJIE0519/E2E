# Train Stepwise Cox Model (AIC-based)

Fits a Cox model and performs backward stepwise selection based on AIC.

## Usage

``` r
stepcox_pro(X, y_surv, tune = FALSE)
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

An object of class `survival_stepcox` and `pro_model`.
