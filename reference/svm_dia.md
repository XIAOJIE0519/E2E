# Train a Support Vector Machine (Linear Kernel) Model for Classification

Trains a Support Vector Machine (SVM) model with a linear kernel using
[`caret::train`](https://rdrr.io/pkg/caret/man/train.html) for binary
classification.

## Usage

``` r
svm_dia(X, y, tune = FALSE, cv_folds = 5)
```

## Arguments

- X:

  A data frame of features.

- y:

  A factor vector of class labels.

- tune:

  Logical, whether to perform hyperparameter tuning using `caret`'s
  default grid (if `TRUE`) or a fixed value (if `FALSE`).

- cv_folds:

  An integer, the number of cross-validation folds for `caret`.

## Value

A [`caret::train`](https://rdrr.io/pkg/caret/man/train.html) object
representing the trained SVM model.

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
svm_model <- svm_dia(X_toy, y_toy)
#> maximum number of iterations reached 0.0001108805 -0.0001108616maximum number of iterations reached 8.843215e-05 -8.840638e-05
print(svm_model)
#> Support Vector Machines with Linear Kernel 
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
#>   0.5666667  0     1   
#> 
#> Tuning parameter 'C' was held constant at a value of 1
# }
```
