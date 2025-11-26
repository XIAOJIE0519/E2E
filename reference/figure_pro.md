# Plot Prognostic Model Evaluation Figures

Generates and returns a ggplot object for Kaplan-Meier (KM) survival
curves or time-dependent ROC curves.

## Usage

``` r
figure_pro(type, data, file = NULL, time_unit = "days")
```

## Arguments

- type:

  "km" or "tdroc"

- data:

  list with:

  - sample_score: data.frame(time, outcome, score)

  - evaluation_metrics: for "km" needs KM_Cutoff; for "tdroc" needs
    AUROC_Years (numeric years like c(1,3,5), OR a named vector/list
    like c('1'=0.74,'3'=0.82,'5'=0.85))

- file:

  optional path to save

- time_unit:

  "days" (default), "months", or "years" for df\$time

## Value

ggplot object
