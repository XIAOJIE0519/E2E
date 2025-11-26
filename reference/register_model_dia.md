# Register a Diagnostic Model Function

Registers a user-defined or pre-defined diagnostic model function with
the internal model registry. This allows the function to be called later
by its registered name, facilitating a modular model management system.

## Usage

``` r
register_model_dia(name, func)
```

## Arguments

- name:

  A character string, the unique name to register the model under.

- func:

  A function, the R function implementing the diagnostic model. This
  function should typically accept `X` (features) and `y` (labels) as
  its first two arguments and return a
  [`caret::train`](https://rdrr.io/pkg/caret/man/train.html) object.

## Value

NULL. The function registers the model function invisibly.

## See also

[`get_registered_models_dia`](https://xiaojie0519.github.io/E2E/reference/get_registered_models_dia.md),
[`initialize_modeling_system_dia`](https://xiaojie0519.github.io/E2E/reference/initialize_modeling_system_dia.md)

## Examples

``` r
# \donttest{
# Example of a dummy model function for registration
my_dummy_rf_model <- function(X, y, tune = FALSE, cv_folds = 5) {
  message("Training dummy RF model...")
  # This is a placeholder and doesn't train a real model.
  # It returns a list with a structure similar to a caret train object.
  list(method = "dummy_rf")
}

# Initialize the system before registering
initialize_modeling_system_dia()
#> Diagnostic modeling system already initialized

# Register the new model
register_model_dia("dummy_rf", my_dummy_rf_model)

# Verify that the model is now in the list of registered models
"dummy_rf" %in% names(get_registered_models_dia())
#> [1] TRUE
# }
```
