# Train a Ridge (L2 Regularized Logistic Regression) Model for Classification

Trains a Ridge-regularized logistic regression model using
[`caret::train`](https://rdrr.io/pkg/caret/man/train.html) (via `glmnet`
method) for binary classification.

## Usage

``` r
ridge_dia(X, y, tune = FALSE, cv_folds = 5)
```

## Arguments

- X:

  A data frame of features.

- y:

  A factor vector of class labels.

- tune:

  Logical, whether to perform hyperparameter tuning for `lambda` (if
  `TRUE`) or use a fixed value (if `FALSE`). `alpha` is fixed at 0 for
  Ridge.

- cv_folds:

  An integer, the number of cross-validation folds for `caret`.

## Value

A [`caret::train`](https://rdrr.io/pkg/caret/man/train.html) object
representing the trained Ridge model.

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
ridge_model <- ridge_dia(X_toy, y_toy)
print(ridge_model)
#> glmnet 
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
#>   0.4666667  0.1   0.9 
#> 
#> Tuning parameter 'alpha' was held constant at a value of 0
#> Tuning
#>  parameter 'lambda' was held constant at a value of 0.01
# }
```
