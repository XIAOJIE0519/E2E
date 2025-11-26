# Initialize Diagnostic Modeling System

Initializes the diagnostic modeling system by loading required packages
and registering default diagnostic models (Random Forest, XGBoost, SVM,
MLP, Lasso, Elastic Net, Ridge, LDA, QDA, Naive Bayes, Decision Tree,
GBM). This function should be called once before using
[`models_dia()`](https://xiaojie0519.github.io/E2E/reference/models_dia.md)
or ensemble methods.

## Usage

``` r
initialize_modeling_system_dia()
```

## Value

Invisible NULL. Initializes the internal model registry.

## Examples

``` r
# \donttest{
# Initialize the system (typically run once at the start of a session or script)
initialize_modeling_system_dia()
#> Diagnostic modeling system already initialized

# Check if a default model like Random Forest is now registered
"rf" %in% names(get_registered_models_dia())
#> [1] TRUE
# }
```
