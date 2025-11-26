# 3. Prognostic Workflow

``` r
library(E2E)
```

## Prognostic Models (Survival Analysis)

This track is dedicated to survival prediction tasks.

### 1. Initialization

First, initialize the prognostic modeling system.

``` r
initialize_modeling_system_pro()
#> Prognosis modeling system initialized.
```

### 2. Training Single Models with `models_pro`

The `models_pro` function trains one or more standard survival models.
For this demonstration, we’ll run a subset.

``` r
# Run a subset of available prognostic models. If all, use model = "all_pro".
results_all_pro <- models_pro(train_pro, model = c("lasso_pro", "rsf_pro"))
#> Running model: lasso_pro
#> Running model: rsf_pro

# Print summary for Random Survival Forest
print_model_summary_pro("rsf_pro", results_all_pro$rsf_pro)
#> 
#> --- rsf_pro Summary ---
#> C-index: 0.9458
#> Avg AUC: 0.9191
#> KM HR: 92.2028 (p=1.116e-17)
```

### 3. Ensemble Modeling

#### Bagging (`bagging_pro`)

Builds a Bagging ensemble for survival models.

``` r
# Create a Bagging ensemble with lasso as the base survival model
# n_estimators is reduced for faster execution.
bagging_lasso_pro_results <- bagging_pro(train_pro, base_model_name = "lasso_pro", n_estimators = 5, seed = 123)
#> Running Bagging: Target 5 models using lasso_pro...
print_model_summary_pro("Bagging (LASSO)", bagging_lasso_pro_results)
#> 
#> --- Bagging (LASSO) Summary ---
#> C-index: 0.7330
#> Avg AUC: 0.5831
#> KM HR: 3.2048 (p=2.22e-08)
```

#### Stacking (`stacking_pro`)

Builds a Stacking ensemble for survival models.

``` r
# Create a Stacking ensemble with lasso as the meta-model
stacking_lasso_pro_results <- stacking_pro(
  results_all_models = results_all_pro,
  data = train_pro,
  meta_model_name = "lasso_pro"
)
print_model_summary_pro("Stacking (LASSO)", stacking_lasso_pro_results)
#> 
#> --- Stacking (LASSO) Summary ---
#> C-index: 0.9491
#> Avg AUC: 0.9243
#> KM HR: 93.2054 (p=8.946e-18)
```

### 4. Applying Models to New Data (`apply_pro`)

Generate prognostic scores for a new dataset.

``` r
# Apply the trained stacking model to the test set
pro_pred_new <- apply_pro(
  trained_model_object = stacking_lasso_pro_results$model_object,
  new_data = test_pro,
  time_unit = "day"
)
#> Applying model on new data...

# Evaluate the new prognostic scores
eval_pro_new <- evaluate_predictions_pro(
  prediction_df = pro_pred_new,
  years_to_evaluate = c(1,3, 5)
)
print(eval_pro_new)
#> $C_index
#> [1] 0.6549641
#> 
#> $AUROC_Years
#> $AUROC_Years$`1`
#> [1] 0.6108769
#> 
#> $AUROC_Years$`3`
#> [1] 0.6592235
#> 
#> $AUROC_Years$`5`
#> [1] 0.569949
#> 
#> 
#> $AUROC_Average
#> [1] 0.6133498
#> 
#> $KM_HR
#> [1] 1.574968
#> 
#> $KM_P_value
#> [1] 0.1311643
#> 
#> $KM_Cutoff
#> [1] 2.228376
```

### 5. Visualization (`figure_pro`)

Generate Kaplan-Meier (KM) and time-dependent ROC (tdROC) curves.

``` r
# Kaplan-Meier Curve
p4 <- figure_pro(type = "km", data = stacking_lasso_pro_results, time_unit= "days")
#print(p4)

# Time-Dependent ROC Curve
p5 <- figure_pro(type = "tdroc", data = stacking_lasso_pro_results, time_unit = "days")
#print(p5)
```
