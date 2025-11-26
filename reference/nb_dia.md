# Train a Naive Bayes Model for Classification

Trains a Naive Bayes model using
[`caret::train`](https://rdrr.io/pkg/caret/man/train.html) for binary
classification.

## Usage

``` r
nb_dia(X, y, tune = FALSE, cv_folds = 5)
```

## Arguments

- X:

  A data frame of features.

- y:

  A factor vector of class labels.

- tune:

  Logical, whether to perform hyperparameter tuning using `caret`'s
  default grid (if `TRUE`) or fixed values (if `FALSE`).

- cv_folds:

  An integer, the number of cross-validation folds for `caret`.

## Value

A [`caret::train`](https://rdrr.io/pkg/caret/man/train.html) object
representing the trained Naive Bayes model.

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
nb_model <- nb_dia(X_toy, y_toy)
#> Registered S3 methods overwritten by 'klaR':
#>   method      from 
#>   predict.rda vegan
#>   print.rda   vegan
#>   plot.rda    vegan
print(nb_model)
#> Naive Bayes 
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
#>   0.3916667  0.1   0.8333333
#> 
#> Tuning parameter 'fL' was held constant at a value of 0
#> Tuning
#>  parameter 'usekernel' was held constant at a value of TRUE
#> Tuning
#>  parameter 'adjust' was held constant at a value of 1
# }
```
