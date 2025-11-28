# Train XGBoost Cox Model

Fits an XGBoost model using the Cox proportional hazards objective
function.

## Usage

``` r
xgb_pro(X, y_surv, tune = FALSE)
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

An object of class `survival_xgboost` and `pro_model`.
