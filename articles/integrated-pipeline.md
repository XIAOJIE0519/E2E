# 0. One-Click Run All Models

``` r
library(E2E)
```

## Overview

E2E provides three powerful one-click functions that automatically run
comprehensive modeling pipelines:

- **[`int_dia()`](https://xiaojie0519.github.io/E2E/reference/int_dia.md)**:
  Diagnostic modeling pipeline (52 model variants)
- **[`int_imbalance()`](https://xiaojie0519.github.io/E2E/reference/int_imbalance.md)**:
  Imbalanced data diagnostic pipeline (64 model variants)
- **[`int_pro()`](https://xiaojie0519.github.io/E2E/reference/int_pro.md)**:
  Prognostic modeling pipeline (24 model variants)

------------------------------------------------------------------------

## 1. Diagnostic Pipeline

``` r
# Run all diagnostic models
results_dia <- int_dia(
  train_dia,
  test_dia,
  test_dia, #can be any other data
  tune = FALSE,
  n_estimators = 5,
  seed = 123
)

# Visualize results
#plot_integrated_results(results_dia, metric_name = "AUROC")
```

------------------------------------------------------------------------

## 2. Imbalanced Data Pipeline

``` r
# Run all models including imbalance handling methods
results_imb <- int_imbalance(
  train_dia,
  test_dia,
  test_dia, #can be any other data
  tune = FALSE,
  n_estimators = 5,
  seed = 123
)

# Visualize results
#plot_integrated_results(results_imb, metric_name = "AUROC")
```

------------------------------------------------------------------------

## 3. Prognostic Pipeline

``` r
# Run all prognostic models
results_pro <- int_pro(
  train_pro,
  test_pro,
  test_pro, #can be any other data
  tune = FALSE,
  n_estimators = 5,
  time_unit = "day",
  years_to_evaluate = c(1, 3, 5),
  seed = 123
)

# Visualize results (C-index)
#plot_integrated_results(results_pro, metric_name = "C-index")
```

------------------------------------------------------------------------

## Key Features

✅ **Fully Automated**: Run dozens of models with one line of code ✅
**Multi-Dataset Evaluation**: Evaluate on training and multiple test
sets simultaneously ✅ **Diverse Methods**: Covers single models,
Bagging, Stacking, and Voting ✅ **Clear Visualization**: Heatmaps
display all results intuitively
