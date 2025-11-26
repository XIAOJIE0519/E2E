# Train a Gradient Boosting Machine (GBM) Model for Classification

Trains a Gradient Boosting Machine (GBM) model using
[`caret::train`](https://rdrr.io/pkg/caret/man/train.html) for binary
classification.

## Usage

``` r
gbm_dia(X, y, tune = FALSE, cv_folds = 5, tune_length = 10)
```

## Arguments

- X:

  A data frame of features.

- y:

  A factor vector of class labels.

- tune:

  Logical, whether to perform hyperparameter tuning for
  `interaction.depth`, `n.trees`, and `shrinkage` (if `TRUE`) or use
  fixed values (if `FALSE`).

- cv_folds:

  An integer, the number of cross-validation folds for `caret`.

- tune_length:

  An integer, the number of random parameter combinations to try when
  tune=TRUE. Only used when search="random". Default is 20.

## Value

A [`caret::train`](https://rdrr.io/pkg/caret/man/train.html) object
representing the trained GBM model.

## Examples

``` r
# \donttest{
set.seed(42)
n_obs <- 200
X_toy <- data.frame(
  FeatureA = rnorm(n_obs),
  FeatureB = runif(n_obs, 0, 100)
)
y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
                levels = c("Control", "Case"))

# Train the model with default parameters
gbm_model <- gbm_dia(X_toy, y_toy)
print(gbm_model)
#> Stochastic Gradient Boosting 
#> 
#> 200 samples
#>   2 predictor
#>   2 classes: 'Control', 'Case' 
#> 
#> No pre-processing
#> Resampling: Cross-Validated (5 fold) 
#> Summary of sample sizes: 161, 159, 160, 160, 160 
#> Resampling results:
#> 
#>   ROC        Sens       Spec     
#>   0.4855489  0.4684211  0.5104762
#> 
#> Tuning parameter 'n.trees' was held constant at a value of 100
#> Tuning
#> 
#> Tuning parameter 'shrinkage' was held constant at a value of 0.1
#> 
#> Tuning parameter 'n.minobsinnode' was held constant at a value of 10

# Train with extensive tuning (random search)
gbm_model_tuned <- gbm_dia(X_toy, y_toy, tune = TRUE, tune_length = 30)
print(gbm_model_tuned)
#> Stochastic Gradient Boosting 
#> 
#> 200 samples
#>   2 predictor
#>   2 classes: 'Control', 'Case' 
#> 
#> No pre-processing
#> Resampling: Cross-Validated (5 fold) 
#> Summary of sample sizes: 159, 161, 160, 160, 160 
#> Resampling results across tuning parameters:
#> 
#>   shrinkage    interaction.depth  n.minobsinnode  n.trees  ROC        Sens     
#>   0.006588957   5                 22               911     0.5445476  0.4668421
#>   0.030901568   3                  7              3064     0.5419649  0.4684211
#>   0.037810451   5                 25              3123     0.4904856  0.4168421
#>   0.057393355   6                 19              2716     0.4979123  0.4489474
#>   0.074852710   9                 22              1224     0.5142544  0.4484211
#>   0.095959811   9                 24              2448     0.4880219  0.4684211
#>   0.099588459   2                 22               356     0.5128440  0.4568421
#>   0.107189977   3                  6              4888     0.5450877  0.4894737
#>   0.114660498  10                 11              4758     0.5181836  0.4678947
#>   0.159841183   7                 12              2409     0.5124330  0.4673684
#>   0.234554175   6                 12              4763     0.5165482  0.4778947
#>   0.244029669   5                 15              4400     0.5135044  0.4800000
#>   0.245543187  10                 11               547     0.4951122  0.4484211
#>   0.278538257   7                  8              3791     0.5368227  0.4389474
#>   0.280722493   2                 20              1829     0.4880883  0.4078947
#>   0.282259344   9                 11              3698     0.5202751  0.4884211
#>   0.313769264   9                 23              4043     0.4784787  0.4289474
#>   0.363386679   3                  6              3482     0.5429536  0.4584211
#>   0.420100644   1                 22              1416     0.4729167  0.4478947
#>   0.425526234   3                 15              2293     0.4935207  0.5078947
#>   0.472165606   8                 23              3273     0.5115182  0.5005263
#>   0.472578043   1                 20               600     0.5119066  0.4600000
#>   0.473236460   7                 18              2809     0.4998346  0.4994737
#>   0.481101330   5                  5              2769     0.5323396  0.3878947
#>   0.482421041   1                 24              3078     0.4773427  0.4584211
#>   0.488062467   2                 22               792     0.4944799  0.4189474
#>   0.488719943   8                 19               557     0.4846372  0.4489474
#>   0.517516988   4                 10              4756     0.5263910  0.6021053
#>   0.558316088   6                  7              2830     0.5411723  0.3668421
#>   0.569471796   1                 11               944     0.5436667  0.4794737
#>   Spec     
#>   0.5395238
#>   0.5200000
#>   0.5409524
#>   0.5009524
#>   0.5600000
#>   0.5800000
#>   0.5595238
#>   0.5980952
#>   0.5200000
#>   0.5004762
#>   0.5590476
#>   0.5109524
#>   0.5300000
#>   0.5980952
#>   0.5495238
#>   0.5195238
#>   0.5690476
#>   0.5871429
#>   0.5204762
#>   0.5180952
#>   0.5209524
#>   0.4709524
#>   0.5100000
#>   0.6276190
#>   0.4619048
#>   0.5595238
#>   0.5200000
#>   0.4109524
#>   0.7061905
#>   0.5490476
#> 
#> ROC was used to select the optimal model using the largest value.
#> The final values used for the model were n.trees = 4888, interaction.depth =
#>  3, shrinkage = 0.10719 and n.minobsinnode = 6.
# }
```
