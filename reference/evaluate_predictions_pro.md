# Evaluate External Predictions

Calculates performance metrics for external prediction sets.

## Usage

``` r
evaluate_predictions_pro(prediction_df, years_to_evaluate = c(1, 3, 5))
```

## Arguments

- prediction_df:

  Data frame with columns `time`, `outcome`, `score`, `ID`.

- years_to_evaluate:

  Years for AUC.

## Value

List of evaluation metrics.
