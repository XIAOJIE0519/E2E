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

## Examples

``` r
# \donttest{
set.seed(42)
n_obs <- 50
X_toy <- data.frame(
  FeatureA = rnorm(n_obs),
  FeatureB = runif(n_obs, 0, 100)
)
y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
                levels = c("Control", "Case"))

# Train the model
xb_model <- xb_dia(X_toy, y_toy)
print(xb_model)
#> eXtreme Gradient Boosting 
#> 
#> 50 samples
#>  2 predictor
#>  2 classes: 'Control', 'Case' 
#> 
#> No pre-processing
#> Resampling: Cross-Validated (5 fold) 
#> Summary of sample sizes: 40, 40, 40, 40, 40 
#> Resampling results:
#> 
#>   ROC        Sens  Spec     
#>   0.5666667  0.45  0.6666667
#> 
#> Tuning parameter 'nrounds' was held constant at a value of 100
#> Tuning
#>  held constant at a value of 1
#> Tuning parameter 'subsample' was held
#>  constant at a value of 1
# }
```
