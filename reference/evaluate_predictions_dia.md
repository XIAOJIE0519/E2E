# Evaluate Predictions from a Data Frame

Evaluates model performance from a data frame of predictions,
calculating metrics like AUROC, AUPRC, F1 score, etc. This function is
designed for use with prediction results, such as the output from
`apply_dia`.

## Usage

``` r
evaluate_predictions_dia(
  prediction_df,
  threshold_choices = "default",
  pos_class = "Positive",
  neg_class = "Negative"
)
```

## Arguments

- prediction_df:

  A data frame containing predictions. Must contain the columns
  `sample`, `label` (true labels), and `score` (predicted
  probabilities).

- threshold_choices:

  A character string specifying the thresholding strategy ("default",
  "f1", "youden") or a numeric probability threshold value (0-1).

- pos_class:

  A character string for the positive class label used in reporting.
  **Defaults to `"Positive"`.**

- neg_class:

  A character string for the negative class label used in reporting.
  **Defaults to `"Negative"`.**

## Value

A named list containing all calculated performance metrics.

## Details

This function strictly requires the `label` column in `prediction_df` to
adhere to the following format:

- **`1`**: Represents the positive class.

- **`0`**: Represents the negative class.

- **`NA`**: Will be ignored during calculation.

The function will stop with an error if any other values are found in
the `label` column.

## Examples

``` r
# \donttest{
# # Create a sample prediction data frame
# predictions_df <- data.frame(
#   sample = 1:10,
#   label = c(1, 0, 1, 1, 0, 0, 1, 0, 1, 0),
#   score = c(0.9, 0.2, 0.8, 0.6, 0.3, 0.4, 0.95, 0.1, 0.7, 0.5)
# )
#
# # Evaluate the predictions using the 'f1' threshold strategy
# evaluation_results <- evaluate_predictions_dia(
#   prediction_df = predictions_df,
#   threshold_choices = "f1"
# )
#
# print(evaluation_results)
# }
```
