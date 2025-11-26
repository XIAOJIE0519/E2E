# Train Stacking Ensemble for Prognosis

Implements a Stacking Ensemble (Super Learner). It uses the risk scores
from top-performing base models as meta-features to train a second-level
meta-learner.

## Usage

``` r
stacking_pro(
  results_all_models,
  data,
  meta_model_name,
  top = 3,
  tune_meta = FALSE,
  time_unit = "day",
  years_to_evaluate = c(1, 3, 5),
  seed = 789
)
```

## Arguments

- results_all_models:

  List of results from
  [`models_pro()`](https://xiaojie0519.github.io/E2E/reference/models_pro.md).

- data:

  Training data.

- meta_model_name:

  Name of the meta-learner (e.g., "lasso_pro").

- top:

  Integer. Number of top base models to include based on C-index.

- tune_meta:

  Logical. Tune the meta-learner?

- time_unit:

  Time unit.

- years_to_evaluate:

  Evaluation years.

- seed:

  Integer seed.

## Value

A list containing the stacking object and evaluation results.
