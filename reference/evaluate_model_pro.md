# Evaluate Prognostic Model Performance

Comprehensive evaluation of survival models using:

1.  Harrell's Concordance Index (C-index).

2.  Time-dependent Area Under the ROC Curve (AUROC) at specified years.

3.  Kaplan-Meier analysis comparing high vs. low risk groups (based on
    median split).

## Usage

``` r
evaluate_model_pro(
  trained_model_obj = NULL,
  X_data = NULL,
  Y_surv_obj,
  sample_ids,
  years_to_evaluate = c(1, 3, 5),
  precomputed_score = NULL,
  meta_normalize_params = NULL
)
```

## Arguments

- trained_model_obj:

  A trained model object (optional if precomputed_score provided).

- X_data:

  Features for prediction (optional if precomputed_score provided).

- Y_surv_obj:

  True survival object.

- sample_ids:

  Vector of IDs.

- years_to_evaluate:

  Numeric vector of years for time-dependent AUC.

- precomputed_score:

  Numeric vector of pre-calculated risk scores.

- meta_normalize_params:

  Internal use.

## Value

A list containing a dataframe of scores and a list of evaluation
metrics.
