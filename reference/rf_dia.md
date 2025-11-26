# Train a Random Forest Model for Classification

Trains a Random Forest model using
[`caret::train`](https://rdrr.io/pkg/caret/man/train.html) for binary
classification.

## Usage

``` r
rf_dia(X, y, tune = FALSE, cv_folds = 5)
```

## Arguments

- X:

  A data frame of features.

- y:

  A factor vector of class labels.

- tune:

  Logical, whether to perform hyperparameter tuning using `caret`'s
  default grid (if `TRUE`) or use a fixed `mtry` value (if `FALSE`).

- cv_folds:

  An integer, the number of cross-validation folds for `caret`.

## Value

A [`caret::train`](https://rdrr.io/pkg/caret/man/train.html) object
representing the trained Random Forest model.

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
rf_model <- rf_dia(X_toy, y_toy)
print(rf_model)
#> Random Forest 
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
#>   0.6041667  0.45  0.6666667
#> 
#> Tuning parameter 'mtry' was held constant at a value of 1
# }
```
