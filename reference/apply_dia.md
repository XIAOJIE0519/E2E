# Apply a Trained Model to New Data

Applies a trained diagnostic model (single or ensemble) to a new dataset
to generate predictions. It can handle various model objects created by
the package, including single caret models, Bagging, Stacking, Voting,
and EasyEnsemble objects.

## Usage

``` r
apply_dia(
  trained_model_object,
  new_data,
  label_col_name = NULL,
  pos_class = "Positive",
  neg_class = "Negative"
)
```

## Arguments

- trained_model_object:

  A trained model object from `models_dia`, `bagging_dia`,
  `stacking_dia`, `voting_dia`, or `imbalance_dia`.

- new_data:

  A data frame containing the new samples for prediction. The first
  column must be the sample ID.

- label_col_name:

  An optional character string specifying the name of the column in
  `new_data` that contains the true labels. **If `NULL` (the default),
  the function will assume the second column is the label column.** To
  explicitly prevent label extraction (e.g., for data without labels),
  provide `NA`.

- pos_class:

  A character string for the positive class label used in the model's
  probability predictions. **Defaults to `"Positive"`.**

- neg_class:

  A character string for the negative class label. This parameter is
  mainly for consistency, as prediction focuses on `pos_class`
  probability. **Defaults to `"Negative"`.**

## Value

A data frame with three columns: `sample` (the sample IDs), `label` (the
true labels from `new_data`, or `NA` if not available/specified), and
`score` (the predicted probability for the positive class).

## Examples

``` r
# \donttest{
# Assuming `bagging_results` and `test_dia` are available from previous steps
# bagging_model <- bagging_results$model_object

# Example 1: Default behavior - use the second column of test_dia as label
# predictions <- apply_dia(
#   trained_model_object = bagging_model,
#   new_data = test_dia
# )

# Example 2: Explicitly specify the label column by name
# predictions_explicit <- apply_dia(
#   trained_model_object = bagging_model,
#   new_data = test_dia,
#   label_col_name = "outcome"
# )

# Example 3: Predict on data without labels
# test_data_no_labels <- test_dia[, -2] # Remove outcome column
# predictions_no_label <- apply_dia(
#   trained_model_object = bagging_model,
#   new_data = test_data_no_labels,
#   label_col_name = NA # Explicitly disable label extraction
# )
# }
```
