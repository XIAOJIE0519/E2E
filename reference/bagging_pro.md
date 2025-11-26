# Train Bagging Ensemble for Prognosis

Implements Bootstrap Aggregating (Bagging) for survival models. It
trains multiple base models on bootstrapped subsets and averages the
risk scores. This method reduces variance and improves stability.

## Usage

``` r
bagging_pro(
  data,
  base_model_name,
  n_estimators = 10,
  subset_fraction = 0.632,
  tune_base_model = FALSE,
  time_unit = "day",
  years_to_evaluate = c(1, 3, 5),
  seed = 456
)
```

## Arguments

- data:

  Input data frame (ID, Status, Time, Features).

- base_model_name:

  Character string name of the base model (e.g., "rsf_pro").

- n_estimators:

  Integer. Number of bootstrap iterations.

- subset_fraction:

  Numeric (0-1). Fraction of data to sample in each iteration.

- tune_base_model:

  Logical. Whether to tune each base model (computationally expensive).

- time_unit:

  Time unit of the input data.

- years_to_evaluate:

  Numeric vector of years for time-dependent AUC evaluation.

- seed:

  Integer seed for reproducibility.

## Value

A list containing the ensemble object, sample scores, and evaluation
metrics.
