# Plot Diagnostic Model Evaluation Figures

Generates and returns a ggplot object for Receiver Operating
Characteristic (ROC) curves, Precision-Recall (PRC) curves, or confusion
matrices.

## Usage

``` r
figure_dia(type, data, file = NULL)
```

## Arguments

- type:

  String, specifies the type of plot to generate. Options are "roc",
  "prc", or "matrix".

- data:

  A list object containing model evaluation results. It must include:

  - `sample_score`: A data frame with "label" (0/1) and "score" columns.

  - `evaluation_metrics`: A list with a "Final_Threshold" or
    "Final_Threshold" value.

- file:

  Optional. A string specifying the path to save the plot (e.g.,
  "plot.png"). If `NULL` (the default), the plot object is returned
  instead of being saved.

## Value

A ggplot object. If the `file` argument is provided, the plot is also
saved to the specified path.

## Examples

``` r
# Create example data for a diagnostic model
external_eval_example_dia <- list(
  sample_score = data.frame(
    ID = paste0("S", 1:100),
    label = sample(c(0, 1), 100, replace = TRUE),
    score = runif(100, 0, 1)
  ),
  evaluation_metrics = list(
    Final_Threshold = 0.53
  )
)

# Generate an ROC curve plot object
roc_plot <- figure_dia(type = "roc", data = external_eval_example_dia)
# To display the plot, simply run:
# print(roc_plot)

# Generate a PRC curve and save it to a temporary file
# tempfile() creates a safe, temporary path as required by CRAN
temp_prc_path <- tempfile(fileext = ".png")
figure_dia(type = "prc", data = external_eval_example_dia, file = temp_prc_path)
#> Plot saved to: /tmp/RtmpKrW4Lv/file1f23395e785.png

# Generate a Confusion Matrix plot
matrix_plot <- figure_dia(type = "matrix", data = external_eval_example_dia)
```
