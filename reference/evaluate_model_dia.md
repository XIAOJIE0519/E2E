# Evaluate Diagnostic Model Performance

Evaluates the performance of a trained diagnostic model using various
metrics relevant to binary classification, including AUROC, AUPRC, and
metrics at an optimal or specified probability threshold.

## Usage

``` r
evaluate_model_dia(
  model_obj = NULL,
  X_data = NULL,
  y_data,
  sample_ids,
  threshold_choices = "default",
  pos_class,
  neg_class,
  precomputed_prob = NULL,
  y_original_numeric = NULL
)
```

## Arguments

- model_obj:

  A trained model object (typically a
  [`caret::train`](https://rdrr.io/pkg/caret/man/train.html) object or a
  list from an ensemble like Bagging). Can be `NULL` if
  `precomputed_prob` is provided.

- X_data:

  A data frame of features corresponding to the data used for
  evaluation. Required if `model_obj` is provided and `precomputed_prob`
  is `NULL`.

- y_data:

  A factor vector of true class labels for the evaluation data.

- sample_ids:

  A vector of sample IDs for the evaluation data.

- threshold_choices:

  A character string specifying the thresholding strategy ("default",
  "f1", "youden") or a numeric probability threshold value (0-1).

- pos_class:

  A character string, the label for the positive class.

- neg_class:

  A character string, the label for the negative class.

- precomputed_prob:

  Optional. A numeric vector of precomputed probabilities for the
  positive class. If provided, `model_obj` and `X_data` are not used for
  score derivation.

- y_original_numeric:

  Optional. The original numeric/character vector of labels. If not
  provided, it's inferred from `y_data` using global `pos_label_value`
  and `neg_label_value`.

## Value

A list containing:

- `sample_score`: A data frame with `sample` (ID), `label` (original
  numeric), and `score` (predicted probability for positive class).

- `evaluation_metrics`: A list of performance metrics:

  - `Threshold_Strategy`: The strategy used for threshold selection.

  - `_Threshold`: The chosen probability threshold.

  - `Accuracy`, `Precision`, `Recall`, `F1`, `Specificity`: Metrics
    calculated at `_Threshold`.

  - `AUROC`: Area Under the Receiver Operating Characteristic curve.

  - `AUROC_95CI_Lower`, `AUROC_95CI_Upper`: 95% confidence interval for
    AUROC.

  - `AUPRC`: Area Under the Precision-Recall curve.

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
ids_toy <- paste0("Sample", 1:n_obs)

# 2. Train a model
rf_model <- rf_dia(X_toy, y_toy)

# 3. Evaluate the model using F1-score optimal threshold
eval_results <- evaluate_model_dia(
  model_obj = rf_model,
  X_data = X_toy,
  y_data = y_toy,
  sample_ids = ids_toy,
  threshold_choices = "f1",
  pos_class = "Case",
  neg_class = "Control"
)
#> Warning: ci.auc() of a ROC curve with AUC == 1 is always 1-1 and can be misleading.
str(eval_results)
#> List of 2
#>  $ sample_score      :'data.frame':  50 obs. of  3 variables:
#>   ..$ sample: chr [1:50] "Sample1" "Sample2" "Sample3" "Sample4" ...
#>   ..$ label : num [1:50] 1 1 0 0 0 0 0 0 0 1 ...
#>   ..$ score : num [1:50] 0.67 0.822 0.256 0.302 0.146 0.104 0.122 0.368 0.142 0.724 ...
#>  $ evaluation_metrics:List of 11
#>   ..$ Threshold_Strategy: chr "f1"
#>   ..$ Final_Threshold   : num 0.644
#>   ..$ Accuracy          : Named num 1
#>   .. ..- attr(*, "names")= chr "Accuracy"
#>   ..$ Precision         : Named num 1
#>   .. ..- attr(*, "names")= chr "Precision"
#>   ..$ Recall            : Named num 1
#>   .. ..- attr(*, "names")= chr "Recall"
#>   ..$ F1                : Named num 1
#>   .. ..- attr(*, "names")= chr "F1"
#>   ..$ Specificity       : Named num 1
#>   .. ..- attr(*, "names")= chr "Specificity"
#>   ..$ AUROC             : 'auc' num 1
#>   .. ..- attr(*, "partial.auc")= logi FALSE
#>   .. ..- attr(*, "percent")= logi FALSE
#>   .. ..- attr(*, "roc")=List of 15
#>   .. .. ..$ percent           : logi FALSE
#>   .. .. ..$ sensitivities     : num [1:43] 1 1 1 1 1 1 1 1 1 1 ...
#>   .. .. ..$ specificities     : num [1:43] 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.5 0.55 ...
#>   .. .. ..$ thresholds        : num [1:43] -Inf 0.091 0.101 0.113 0.132 ...
#>   .. .. ..$ direction         : chr "<"
#>   .. .. ..$ cases             : num [1:30] 0.67 0.822 0.724 0.644 0.982 0.896 0.922 0.92 0.782 0.938 ...
#>   .. .. ..$ controls          : num [1:20] 0.256 0.302 0.146 0.104 0.122 0.368 0.142 0.292 0.282 0.238 ...
#>   .. .. ..$ fun.sesp          :function (...)  
#>   .. .. ..$ auc               : 'auc' num 1
#>   .. .. .. ..- attr(*, "partial.auc")= logi FALSE
#>   .. .. .. ..- attr(*, "percent")= logi FALSE
#>   .. .. .. ..- attr(*, "roc")=List of 15
#>   .. .. .. .. ..$ percent           : logi FALSE
#>   .. .. .. .. ..$ sensitivities     : num [1:43] 1 1 1 1 1 1 1 1 1 1 ...
#>   .. .. .. .. ..$ specificities     : num [1:43] 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.5 0.55 ...
#>   .. .. .. .. ..$ thresholds        : num [1:43] -Inf 0.091 0.101 0.113 0.132 ...
#>   .. .. .. .. ..$ direction         : chr "<"
#>   .. .. .. .. ..$ cases             : num [1:30] 0.67 0.822 0.724 0.644 0.982 0.896 0.922 0.92 0.782 0.938 ...
#>   .. .. .. .. ..$ controls          : num [1:20] 0.256 0.302 0.146 0.104 0.122 0.368 0.142 0.292 0.282 0.238 ...
#>   .. .. .. .. ..$ fun.sesp          :function (...)  
#>   .. .. .. .. ..$ auc               : 'auc' num 1
#>   .. .. .. .. .. ..- attr(*, "partial.auc")= logi FALSE
#>   .. .. .. .. .. ..- attr(*, "percent")= logi FALSE
#>   .. .. .. .. .. ..- attr(*, "roc")=List of 8
#>   .. .. .. .. .. .. ..$ percent      : logi FALSE
#>   .. .. .. .. .. .. ..$ sensitivities: num [1:43] 1 1 1 1 1 1 1 1 1 1 ...
#>   .. .. .. .. .. .. ..$ specificities: num [1:43] 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.5 0.55 ...
#>   .. .. .. .. .. .. ..$ thresholds   : num [1:43] -Inf 0.091 0.101 0.113 0.132 ...
#>   .. .. .. .. .. .. ..$ direction    : chr "<"
#>   .. .. .. .. .. .. ..$ cases        : num [1:30] 0.67 0.822 0.724 0.644 0.982 0.896 0.922 0.92 0.782 0.938 ...
#>   .. .. .. .. .. .. ..$ controls     : num [1:20] 0.256 0.302 0.146 0.104 0.122 0.368 0.142 0.292 0.282 0.238 ...
#>   .. .. .. .. .. .. ..$ fun.sesp     :function (...)  
#>   .. .. .. .. .. .. ..- attr(*, "class")= chr "roc"
#>   .. .. .. .. ..$ call              : language roc.default(response = y_data, predictor = prob, levels = c(neg_class,      pos_class), quiet = TRUE)
#>   .. .. .. .. ..$ original.predictor: num [1:50] 0.67 0.822 0.256 0.302 0.146 0.104 0.122 0.368 0.142 0.724 ...
#>   .. .. .. .. ..$ original.response : Factor w/ 2 levels "Control","Case": 2 2 1 1 1 1 1 1 1 2 ...
#>   .. .. .. .. ..$ predictor         : num [1:50] 0.67 0.822 0.256 0.302 0.146 0.104 0.122 0.368 0.142 0.724 ...
#>   .. .. .. .. ..$ response          : Factor w/ 2 levels "Control","Case": 2 2 1 1 1 1 1 1 1 2 ...
#>   .. .. .. .. ..$ levels            : chr [1:2] "Control" "Case"
#>   .. .. .. .. ..- attr(*, "class")= chr "roc"
#>   .. .. ..$ call              : language roc.default(response = y_data, predictor = prob, levels = c(neg_class,      pos_class), quiet = TRUE)
#>   .. .. ..$ original.predictor: num [1:50] 0.67 0.822 0.256 0.302 0.146 0.104 0.122 0.368 0.142 0.724 ...
#>   .. .. ..$ original.response : Factor w/ 2 levels "Control","Case": 2 2 1 1 1 1 1 1 1 2 ...
#>   .. .. ..$ predictor         : num [1:50] 0.67 0.822 0.256 0.302 0.146 0.104 0.122 0.368 0.142 0.724 ...
#>   .. .. ..$ response          : Factor w/ 2 levels "Control","Case": 2 2 1 1 1 1 1 1 1 2 ...
#>   .. .. ..$ levels            : chr [1:2] "Control" "Case"
#>   .. .. ..- attr(*, "class")= chr "roc"
#>   ..$ AUROC_95CI_Lower  : num 1
#>   ..$ AUROC_95CI_Upper  : num 1
#>   ..$ AUPRC             : num 1
# }
```
