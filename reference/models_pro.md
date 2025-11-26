# Run Multiple Prognostic Models

High-level API to train and evaluate multiple survival models in batch.

## Usage

``` r
models_pro(
  data,
  model = "all_pro",
  tune = FALSE,
  seed = 123,
  time_unit = "day",
  years_to_evaluate = c(1, 3, 5)
)
```

## Arguments

- data:

  Input data frame.

- model:

  Character vector of model names or "all_pro".

- tune:

  Logical. Enable hyperparameter tuning?

- seed:

  Random seed.

- time_unit:

  Time unit of input.

- years_to_evaluate:

  Years for AUC calculation.

## Value

A list of model results.
