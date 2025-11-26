# Train Lasso Cox Proportional Hazards Model

Fits a Cox proportional hazards model regularized by the Lasso (L1)
penalty. Uses cross-validation to select the optimal lambda.

## Usage

``` r
lasso_pro(X, y_surv, tune = FALSE)
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

## Examples

``` r
# \donttest{
  library(survival)
#> 
#> Attaching package: ‘survival’
#> The following object is masked from ‘package:caret’:
#> 
#>     cluster
  # Create dummy data
  set.seed(123)
  df <- data.frame(time = rexp(50), status = sample(0:1, 50, replace=TRUE),
                   var1 = rnorm(50), var2 = rnorm(50))
  y <- Surv(df$time, df$status)
  x <- df[, c("var1", "var2")]

  model <- lasso_pro(x, y)
  print(class(model))
#> [1] "survival_glmnet" "pro_model"      
# }
```
