# Train an XGBoost Tree Model for Classification

Trains an Extreme Gradient Boosting (XGBoost) model using
[`caret::train`](https://rdrr.io/pkg/caret/man/train.html) for binary
classification.

## Usage

``` r
xb_dia(X, y, tune = FALSE, cv_folds = 5, tune_length = 20)
```

## Arguments

- X:

  A data frame of features.

- y:

  A factor vector of class labels.

- tune:

  Logical, whether to perform hyperparameter tuning using `caret`'s
  default grid (if `TRUE`) or use fixed values (if `FALSE`).

- cv_folds:

  An integer, the number of cross-validation folds for `caret`.

- tune_length:

  An integer, the number of random parameter combinations to try when
  tune=TRUE. Only used when search="random". Default is 20.

## Value

A [`caret::train`](https://rdrr.io/pkg/caret/man/train.html) object
representing the trained XGBoost model.
