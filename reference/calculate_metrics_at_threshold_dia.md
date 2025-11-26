# Calculate Classification Metrics at a Specific Threshold

Calculates various classification performance metrics (Accuracy,
Precision, Recall, F1-score, Specificity, True Positives, etc.) for
binary classification at a given probability threshold.

## Usage

``` r
calculate_metrics_at_threshold_dia(
  prob_positive,
  y_true,
  threshold,
  pos_class,
  neg_class
)
```

## Arguments

- prob_positive:

  A numeric vector of predicted probabilities for the positive class.

- y_true:

  A factor vector of true class labels.

- threshold:

  A numeric value between 0 and 1, the probability threshold above which
  a prediction is considered positive.

- pos_class:

  A character string, the label for the positive class.

- neg_class:

  A character string, the label for the negative class.

## Value

A list containing:

- `Threshold`: The threshold used.

- `Accuracy`: Overall prediction accuracy.

- `Precision`: Precision for the positive class.

- `Recall`: Recall (Sensitivity) for the positive class.

- `F1`: F1-score for the positive class.

- `Specificity`: Specificity for the negative class.

- `TP`, `TN`, `FP`, `FN`, `N`: Counts of True Positives, True Negatives,
  False Positives, False Negatives, and total samples.

## Examples

``` r
y_true_ex <- factor(c("Negative", "Positive", "Positive", "Negative", "Positive"),
                    levels = c("Negative", "Positive"))
prob_ex <- c(0.1, 0.8, 0.6, 0.3, 0.9)
metrics <- calculate_metrics_at_threshold_dia(
  prob_positive = prob_ex,
  y_true = y_true_ex,
  threshold = 0.5,
  pos_class = "Positive",
  neg_class = "Negative"
)
print(metrics)
#> $Threshold
#> [1] 0.5
#> 
#> $Accuracy
#> Accuracy 
#>        1 
#> 
#> $Precision
#> Precision 
#>         1 
#> 
#> $Recall
#> Recall 
#>      1 
#> 
#> $F1
#> F1 
#>  1 
#> 
#> $Specificity
#> Specificity 
#>           1 
#> 
#> $TP
#> [1] 3
#> 
#> $TN
#> [1] 2
#> 
#> $FP
#> [1] 0
#> 
#> $FN
#> [1] 0
#> 
#> $N
#> [1] 5
#> 
```
