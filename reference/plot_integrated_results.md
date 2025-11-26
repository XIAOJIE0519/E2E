# Visualize Integrated Modeling Results

Creates a heatmap visualization with performance metrics across models
and datasets, including category annotations and summary bar plots.

## Usage

``` r
plot_integrated_results(results_obj, metric_name = "AUROC", output_file = NULL)
```

## Arguments

- results_obj:

  Output from `int_dia`, `int_imbalance`, or `int_pro`.

- metric_name:

  Character string for metric used (e.g., "AUROC", "C-index").

- output_file:

  Optional file path to save plot. If NULL, plot is displayed.

## Value

A ggplot object (invisibly).

## Examples

``` r
if (FALSE) { # \dontrun{
results <- int_dia(train_dia, test_dia)
plot_integrated_results(results, "AUROC")
} # }
```
