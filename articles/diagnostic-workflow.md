# 2. Diagnostic Workflow

``` r
library(E2E)
```

## Diagnostic Models (Classification)

This track is dedicated to binary classification tasks.

### 1. Initialization

First, initialize the diagnostic modeling system. This registers all
built-in classification models.

``` r
initialize_modeling_system_dia()
#> Diagnostic modeling system initialized and default models registered.
```

### 2. Training Single Models with `models_dia`

The `models_dia` function is the gateway to training one or more
standard classification models.

##### Basic Usage

By default, `models_dia` runs all registered models. For this
demonstration, we’ll run a subset to save time.

``` r
# To run all, use model = "all_dia".
results_all_dia <- models_dia(train_dia, model = c("rf", "lasso", "xb"))
#> Running model: rf
#> Warning in ci.auc.roc(roc_obj, conf.level = 0.95): ci.auc() of a ROC curve with
#> AUC == 1 is always 1-1 and can be misleading.
#> Running model: lasso
#> Running model: xb
#> Warning in ci.auc.roc(roc_obj, conf.level = 0.95): ci.auc() of a ROC curve with
#> AUC == 1 is always 1-1 and can be misleading.

# Print a summary for a specific model (e.g., Random Forest)
print_model_summary_dia("rf", results_all_dia$rf)
#> 
#> --- rf Model (on Training Data) Metrics ---
#> Threshold Strategy: default (0.5000)
#> AUROC: 1.0000 (95% CI: 1.0000 - 1.0000)
#> AUPRC: 1.0000
#> Accuracy: 1.0000
#> F1: 1.0000
#> Precision: 1.0000
#> Recall: 1.0000
#> Specificity: 1.0000
#> --------------------------------------------------
```

##### Advanced Usage & Customization

You can precisely control the modeling process by specifying parameters.

``` r
# Run a specific subset of models with tuning enabled and custom thresholds
results_dia_custom <- models_dia(
  data = train_dia,
  model = c("rf", "lasso", "xb"),
  tune = TRUE,
  seed = 123,
  threshold_choices = list(rf = "f1", lasso = 0.6, xb = "youden")
)
#> Running model: rf
#> Warning in ci.auc.roc(roc_obj, conf.level = 0.95): ci.auc() of a ROC curve with
#> AUC == 1 is always 1-1 and can be misleading.
#> Running model: lasso
#> Running model: xb

# View the custom results
print_model_summary_dia("rf", results_dia_custom$rf)
#> 
#> --- rf Model (on Training Data) Metrics ---
#> Threshold Strategy: f1 (0.6220)
#> AUROC: 1.0000 (95% CI: 1.0000 - 1.0000)
#> AUPRC: 1.0000
#> Accuracy: 1.0000
#> F1: 1.0000
#> Precision: 1.0000
#> Recall: 1.0000
#> Specificity: 1.0000
#> --------------------------------------------------
```

### 3. Ensemble Modeling

#### Bagging (`bagging_dia`)

Builds a Bagging ensemble by training a base model on multiple bootstrap
samples.

``` r
# Create a Bagging ensemble with XGBoost as the base model
# n_estimators is reduced for faster execution in this example.
bagging_xb_results <- bagging_dia(train_dia, base_model_name = "xb",
                                  tune_base_model = FALSE, n_estimators = 5)
#> Running Bagging model: Bagging_dia (base: xb)
print_model_summary_dia("Bagging (XGBoost)", bagging_xb_results)
#> 
#> --- Bagging (XGBoost) Model (on Training Data) Metrics ---
#> Ensemble Type: Bagging (Base: xb, Estimators: 5)
#> Threshold Strategy: default (0.5000)
#> AUROC: 0.9995 (95% CI: 0.9989 - 1.0000)
#> AUPRC: 0.9999
#> Accuracy: 0.9919
#> F1: 0.9955
#> Precision: 0.9911
#> Recall: 1.0000
#> Specificity: 0.9125
#> --------------------------------------------------
```

#### Voting (`voting_dia`)

Combines predictions from multiple pre-trained models.

``` r
# Create a soft voting ensemble from the top models
voting_soft_results <- voting_dia(
  results_all_models = results_all_dia,
  data = train_dia,
  type = "soft"
)
#> Running Voting model: Voting_dia (type: soft)
#> Warning in ci.auc.roc(roc_obj, conf.level = 0.95): ci.auc() of a ROC curve with
#> AUC == 1 is always 1-1 and can be misleading.
print_model_summary_dia("Voting (Soft)", voting_soft_results)
#> 
#> --- Voting (Soft) Model (on Training Data) Metrics ---
#> Ensemble Type: Voting (Type: soft, Weight Metric: AUROC, Base models used: rf, xb, lasso)
#> Threshold Strategy: f1 (0.6027)
#> AUROC: 1.0000 (95% CI: 1.0000 - 1.0000)
#> AUPRC: 1.0000
#> Accuracy: 1.0000
#> F1: 1.0000
#> Precision: 1.0000
#> Recall: 1.0000
#> Specificity: 1.0000
#> --------------------------------------------------
```

#### Stacking (`stacking_dia`)

Uses predictions from base models as features to train a final
meta-model.

``` r
# Create a Stacking ensemble with Lasso as the meta-model
stacking_lasso_results <- stacking_dia(
  results_all_models = results_all_dia,
  data = train_dia,
  meta_model_name = "lasso"
)
#> Running Stacking model: Stacking_dia (meta: lasso)
#> Warning in ci.auc.roc(roc_obj, conf.level = 0.95): ci.auc() of a ROC curve with
#> AUC == 1 is always 1-1 and can be misleading.
print_model_summary_dia("Stacking (Lasso)", stacking_lasso_results)
#> 
#> --- Stacking (Lasso) Model (on Training Data) Metrics ---
#> Ensemble Type: Stacking (Meta: lasso, Base models used: rf, xb, lasso)
#> Threshold Strategy: f1 (0.9794)
#> AUROC: 1.0000 (95% CI: 1.0000 - 1.0000)
#> AUPRC: 1.0000
#> Accuracy: 1.0000
#> F1: 1.0000
#> Precision: 1.0000
#> Recall: 1.0000
#> Specificity: 1.0000
#> --------------------------------------------------
```

#### Handling Imbalanced Data (`imbalance_dia`)

Implements the EasyEnsemble algorithm.

``` r
# Create an EasyEnsemble with XGBoost as the base model
# n_estimators is reduced for faster execution.
results_imbalance_dia <- imbalance_dia(train_dia, base_model_name = "xb",
                                       tune_base_model = FALSE, n_estimators = 5)
#> Running Imbalance model: EasyEnsemble_dia (base: xb)
print_model_summary_dia("Imbalance (XGBoost)", results_imbalance_dia)
#> 
#> --- Imbalance (XGBoost) Model (on Training Data) Metrics ---
#> Ensemble Type: EasyEnsemble (Base: xb, Estimators: 5)
#> Threshold Strategy: default (0.5000)
#> AUROC: 0.9992 (95% CI: 0.9976 - 1.0000)
#> AUPRC: 0.9999
#> Accuracy: 0.9768
#> F1: 0.9871
#> Precision: 1.0000
#> Recall: 0.9745
#> Specificity: 1.0000
#> --------------------------------------------------
```

### 4. Applying Models to New Data (`apply_dia`)

Use a trained model object to make predictions on a new, unseen dataset.

``` r
# Apply the trained Bagging model to the test set
bagging_pred_new <- apply_dia(
  trained_model_object = bagging_xb_results$model_object,
  new_data = test_dia,
  label_col_name = "outcome"
)

# Evaluate these new predictions
eval_results_new <- evaluate_predictions_dia(
  prediction_df = bagging_pred_new,
  threshold_choices = "f1")
print(eval_results_new)
#> $Threshold_Strategy
#> [1] "f1"
#> 
#> $Threshold
#> [1] 0.2747457
#> 
#> $Accuracy
#>  Accuracy 
#> 0.9836957 
#> 
#> $Precision
#> Precision 
#> 0.9824047 
#> 
#> $Recall
#> Recall 
#>      1 
#> 
#> $F1
#>        F1 
#> 0.9911243 
#> 
#> $Specificity
#> Specificity 
#>   0.8181818 
#> 
#> $AUROC
#> [1] 0.9977386
#> 
#> $AUROC_95CI_Lower
#> [1] 0.9951056
#> 
#> $AUROC_95CI_Upper
#> [1] 1
#> 
#> $AUPRC
#> [1] 0.9997789
```

### 5. Visualization (`figure_dia`)

Generate high-quality plots to evaluate model performance.

``` r
# ROC Curve
p1 <- figure_dia(type = "roc", data = results_imbalance_dia)
#plot(p1)

# Precision-Recall Curve
p2 <- figure_dia(type = "prc", data = results_imbalance_dia)
#plot(p2)

# Confusion Matrix
p3 <- figure_dia(type = "matrix", data = results_imbalance_dia)
#plot(p3)
```
