# Get Registered Diagnostic Models

Retrieves a list of all diagnostic model functions currently registered
in the internal environment.

## Usage

``` r
get_registered_models_dia()
```

## Value

A named list where names are the registered model names and values are
the corresponding model functions.

## See also

[`register_model_dia`](https://xiaojie0519.github.io/E2E/reference/register_model_dia.md),
[`initialize_modeling_system_dia`](https://xiaojie0519.github.io/E2E/reference/initialize_modeling_system_dia.md)

## Examples

``` r
# \donttest{
# Ensure system is initialized to see the default models
initialize_modeling_system_dia()
#> Diagnostic modeling system already initialized
models <- get_registered_models_dia()
# See available model names
print(names(models))
#>  [1] "rf"    "xb"    "svm"   "mlp"   "lasso" "en"    "ridge" "lda"   "qda"  
#> [10] "nb"    "dt"    "gbm"  
# }
```
