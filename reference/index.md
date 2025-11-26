# Package index

## Integrated Pipelines

One-click functions for comprehensive model comparison across multiple
algorithms and ensemble methods.

- [`int_dia()`](https://xiaojie0519.github.io/E2E/reference/int_dia.md)
  : Comprehensive Diagnostic Modeling Pipeline
- [`int_imbalance()`](https://xiaojie0519.github.io/E2E/reference/int_imbalance.md)
  : Imbalanced Data Diagnostic Modeling Pipeline
- [`int_pro()`](https://xiaojie0519.github.io/E2E/reference/int_pro.md)
  : Comprehensive Prognostic Modeling Pipeline
- [`plot_integrated_results()`](https://xiaojie0519.github.io/E2E/reference/plot_integrated_results.md)
  : Visualize Integrated Modeling Results

## Core Modeling Functions

Main functions for building and evaluating models.

- [`models_dia()`](https://xiaojie0519.github.io/E2E/reference/models_dia.md)
  : Run Multiple Diagnostic Models
- [`models_pro()`](https://xiaojie0519.github.io/E2E/reference/models_pro.md)
  : Run Multiple Prognostic Models
- [`bagging_dia()`](https://xiaojie0519.github.io/E2E/reference/bagging_dia.md)
  : Train a Bagging Diagnostic Model
- [`bagging_pro()`](https://xiaojie0519.github.io/E2E/reference/bagging_pro.md)
  : Train Bagging Ensemble for Prognosis
- [`voting_dia()`](https://xiaojie0519.github.io/E2E/reference/voting_dia.md)
  : Train a Voting Ensemble Diagnostic Model
- [`stacking_dia()`](https://xiaojie0519.github.io/E2E/reference/stacking_dia.md)
  : Train a Stacking Diagnostic Model
- [`stacking_pro()`](https://xiaojie0519.github.io/E2E/reference/stacking_pro.md)
  : Train Stacking Ensemble for Prognosis
- [`imbalance_dia()`](https://xiaojie0519.github.io/E2E/reference/imbalance_dia.md)
  : Train an EasyEnsemble Model for Imbalanced Classification

## Helpers & Visualization

Functions for applying, evaluating, visualizing, and explaining models.

- [`apply_dia()`](https://xiaojie0519.github.io/E2E/reference/apply_dia.md)
  : Apply a Trained Model to New Data
- [`apply_pro()`](https://xiaojie0519.github.io/E2E/reference/apply_pro.md)
  : Apply Prognostic Model to New Data
- [`evaluate_model_dia()`](https://xiaojie0519.github.io/E2E/reference/evaluate_model_dia.md)
  : Evaluate Diagnostic Model Performance
- [`evaluate_model_pro()`](https://xiaojie0519.github.io/E2E/reference/evaluate_model_pro.md)
  : Evaluate Prognostic Model Performance
- [`evaluate_predictions_dia()`](https://xiaojie0519.github.io/E2E/reference/evaluate_predictions_dia.md)
  : Evaluate Predictions from a Data Frame
- [`evaluate_predictions_pro()`](https://xiaojie0519.github.io/E2E/reference/evaluate_predictions_pro.md)
  : Evaluate External Predictions
- [`figure_dia()`](https://xiaojie0519.github.io/E2E/reference/figure_dia.md)
  : Plot Diagnostic Model Evaluation Figures
- [`figure_pro()`](https://xiaojie0519.github.io/E2E/reference/figure_pro.md)
  : Plot Prognostic Model Evaluation Figures
- [`figure_shap()`](https://xiaojie0519.github.io/E2E/reference/figure_shap.md)
  : Generate and Plot SHAP Explanation Figures
- [`print_model_summary_dia()`](https://xiaojie0519.github.io/E2E/reference/print_model_summary_dia.md)
  : Print Diagnostic Model Summary
- [`print_model_summary_pro()`](https://xiaojie0519.github.io/E2E/reference/print_model_summary_pro.md)
  : Print Prognostic Model Summary

## Setup & Customization

Functions for initialization and framework extension.

- [`initialize_modeling_system_dia()`](https://xiaojie0519.github.io/E2E/reference/initialize_modeling_system_dia.md)
  : Initialize Diagnostic Modeling System
- [`initialize_modeling_system_pro()`](https://xiaojie0519.github.io/E2E/reference/initialize_modeling_system_pro.md)
  : Initialize Prognosis Modeling System
- [`register_model_dia()`](https://xiaojie0519.github.io/E2E/reference/register_model_dia.md)
  : Register a Diagnostic Model Function
- [`register_model_pro()`](https://xiaojie0519.github.io/E2E/reference/register_model_pro.md)
  : Register a Prognostic Model
- [`get_registered_models_dia()`](https://xiaojie0519.github.io/E2E/reference/get_registered_models_dia.md)
  : Get Registered Diagnostic Models
- [`get_registered_models_pro()`](https://xiaojie0519.github.io/E2E/reference/get_registered_models_pro.md)
  : Get Registered Prognostic Models

## Datasets

Example datasets included with the package.

- [`train_dia`](https://xiaojie0519.github.io/E2E/reference/train_dia.md)
  : Training Data for Diagnostic Models
- [`test_dia`](https://xiaojie0519.github.io/E2E/reference/test_dia.md)
  : Test Data for Diagnostic Models
- [`train_pro`](https://xiaojie0519.github.io/E2E/reference/train_pro.md)
  : Training Data for Prognostic (Survival) Models
- [`test_pro`](https://xiaojie0519.github.io/E2E/reference/test_pro.md)
  : Test Data for Prognostic (Survival) Models

## Internal & Component Functions

These are lower-level functions, generally not called directly by the
user.

- [`apply_dia()`](https://xiaojie0519.github.io/E2E/reference/apply_dia.md)
  : Apply a Trained Model to New Data
- [`apply_pro()`](https://xiaojie0519.github.io/E2E/reference/apply_pro.md)
  : Apply Prognostic Model to New Data
- [`bagging_dia()`](https://xiaojie0519.github.io/E2E/reference/bagging_dia.md)
  : Train a Bagging Diagnostic Model
- [`bagging_pro()`](https://xiaojie0519.github.io/E2E/reference/bagging_pro.md)
  : Train Bagging Ensemble for Prognosis
- [`calculate_metrics_at_threshold_dia()`](https://xiaojie0519.github.io/E2E/reference/calculate_metrics_at_threshold_dia.md)
  : Calculate Classification Metrics at a Specific Threshold
- [`cb_pro()`](https://xiaojie0519.github.io/E2E/reference/cb_pro.md) :
  Train CoxBoost
- [`dt_dia()`](https://xiaojie0519.github.io/E2E/reference/dt_dia.md) :
  Train a Decision Tree Model for Classification
- [`en_dia()`](https://xiaojie0519.github.io/E2E/reference/en_dia.md) :
  Train an Elastic Net (L1 and L2 Regularized Logistic Regression) Model
  for Classification
- [`en_pro()`](https://xiaojie0519.github.io/E2E/reference/en_pro.md) :
  Train Elastic Net Cox Model
- [`evaluate_model_dia()`](https://xiaojie0519.github.io/E2E/reference/evaluate_model_dia.md)
  : Evaluate Diagnostic Model Performance
- [`evaluate_model_pro()`](https://xiaojie0519.github.io/E2E/reference/evaluate_model_pro.md)
  : Evaluate Prognostic Model Performance
- [`evaluate_predictions_dia()`](https://xiaojie0519.github.io/E2E/reference/evaluate_predictions_dia.md)
  : Evaluate Predictions from a Data Frame
- [`evaluate_predictions_pro()`](https://xiaojie0519.github.io/E2E/reference/evaluate_predictions_pro.md)
  : Evaluate External Predictions
- [`figure_dia()`](https://xiaojie0519.github.io/E2E/reference/figure_dia.md)
  : Plot Diagnostic Model Evaluation Figures
- [`figure_pro()`](https://xiaojie0519.github.io/E2E/reference/figure_pro.md)
  : Plot Prognostic Model Evaluation Figures
- [`find_optimal_threshold_dia()`](https://xiaojie0519.github.io/E2E/reference/find_optimal_threshold_dia.md)
  : Find Optimal Probability Threshold
- [`gbm_dia()`](https://xiaojie0519.github.io/E2E/reference/gbm_dia.md)
  : Train a Gradient Boosting Machine (GBM) Model for Classification
- [`gbm_pro()`](https://xiaojie0519.github.io/E2E/reference/gbm_pro.md)
  : Train Gradient Boosting Machine (GBM) for Survival
- [`get_registered_models_dia()`](https://xiaojie0519.github.io/E2E/reference/get_registered_models_dia.md)
  : Get Registered Diagnostic Models
- [`get_registered_models_pro()`](https://xiaojie0519.github.io/E2E/reference/get_registered_models_pro.md)
  : Get Registered Prognostic Models
- [`imbalance_dia()`](https://xiaojie0519.github.io/E2E/reference/imbalance_dia.md)
  : Train an EasyEnsemble Model for Imbalanced Classification
- [`initialize_modeling_system_dia()`](https://xiaojie0519.github.io/E2E/reference/initialize_modeling_system_dia.md)
  : Initialize Diagnostic Modeling System
- [`initialize_modeling_system_pro()`](https://xiaojie0519.github.io/E2E/reference/initialize_modeling_system_pro.md)
  : Initialize Prognosis Modeling System
- [`int_dia()`](https://xiaojie0519.github.io/E2E/reference/int_dia.md)
  : Comprehensive Diagnostic Modeling Pipeline
- [`int_pro()`](https://xiaojie0519.github.io/E2E/reference/int_pro.md)
  : Comprehensive Prognostic Modeling Pipeline
- [`lasso_dia()`](https://xiaojie0519.github.io/E2E/reference/lasso_dia.md)
  : Train a Lasso (L1 Regularized Logistic Regression) Model for
  Classification
- [`lasso_pro()`](https://xiaojie0519.github.io/E2E/reference/lasso_pro.md)
  : Train Lasso Cox Proportional Hazards Model
- [`lda_dia()`](https://xiaojie0519.github.io/E2E/reference/lda_dia.md)
  : Train a Linear Discriminant Analysis (LDA) Model for Classification
- [`load_and_prepare_data_dia()`](https://xiaojie0519.github.io/E2E/reference/load_and_prepare_data_dia.md)
  : Load and Prepare Data for Diagnostic Models
- [`mlp_dia()`](https://xiaojie0519.github.io/E2E/reference/mlp_dia.md)
  : Train a Multi-Layer Perceptron (Neural Network) Model for
  Classification
- [`models_dia()`](https://xiaojie0519.github.io/E2E/reference/models_dia.md)
  : Run Multiple Diagnostic Models
- [`models_pro()`](https://xiaojie0519.github.io/E2E/reference/models_pro.md)
  : Run Multiple Prognostic Models
- [`nb_dia()`](https://xiaojie0519.github.io/E2E/reference/nb_dia.md) :
  Train a Naive Bayes Model for Classification
- [`pls_pro()`](https://xiaojie0519.github.io/E2E/reference/pls_pro.md)
  : Train Partial Least Squares Cox (PLS-Cox)
- [`predict_pro()`](https://xiaojie0519.github.io/E2E/reference/predict_pro.md)
  : Generic Prediction Interface for Prognostic Models
- [`print_model_summary_dia()`](https://xiaojie0519.github.io/E2E/reference/print_model_summary_dia.md)
  : Print Diagnostic Model Summary
- [`print_model_summary_pro()`](https://xiaojie0519.github.io/E2E/reference/print_model_summary_pro.md)
  : Print Prognostic Model Summary
- [`qda_dia()`](https://xiaojie0519.github.io/E2E/reference/qda_dia.md)
  : Train a Quadratic Discriminant Analysis (QDA) Model for
  Classification
- [`register_model_dia()`](https://xiaojie0519.github.io/E2E/reference/register_model_dia.md)
  : Register a Diagnostic Model Function
- [`register_model_pro()`](https://xiaojie0519.github.io/E2E/reference/register_model_pro.md)
  : Register a Prognostic Model
- [`rf_dia()`](https://xiaojie0519.github.io/E2E/reference/rf_dia.md) :
  Train a Random Forest Model for Classification
- [`ridge_dia()`](https://xiaojie0519.github.io/E2E/reference/ridge_dia.md)
  : Train a Ridge (L2 Regularized Logistic Regression) Model for
  Classification
- [`ridge_pro()`](https://xiaojie0519.github.io/E2E/reference/ridge_pro.md)
  : Train Ridge Cox Model
- [`rsf_pro()`](https://xiaojie0519.github.io/E2E/reference/rsf_pro.md)
  : Train Random Survival Forest (RSF)
- [`stacking_dia()`](https://xiaojie0519.github.io/E2E/reference/stacking_dia.md)
  : Train a Stacking Diagnostic Model
- [`stacking_pro()`](https://xiaojie0519.github.io/E2E/reference/stacking_pro.md)
  : Train Stacking Ensemble for Prognosis
- [`stepcox_pro()`](https://xiaojie0519.github.io/E2E/reference/stepcox_pro.md)
  : Train Stepwise Cox Model (AIC-based)
- [`svm_dia()`](https://xiaojie0519.github.io/E2E/reference/svm_dia.md)
  : Train a Support Vector Machine (Linear Kernel) Model for
  Classification
- [`test_dia`](https://xiaojie0519.github.io/E2E/reference/test_dia.md)
  : Test Data for Diagnostic Models
- [`test_pro`](https://xiaojie0519.github.io/E2E/reference/test_pro.md)
  : Test Data for Prognostic (Survival) Models
- [`train_dia`](https://xiaojie0519.github.io/E2E/reference/train_dia.md)
  : Training Data for Diagnostic Models
- [`train_pro`](https://xiaojie0519.github.io/E2E/reference/train_pro.md)
  : Training Data for Prognostic (Survival) Models
- [`voting_dia()`](https://xiaojie0519.github.io/E2E/reference/voting_dia.md)
  : Train a Voting Ensemble Diagnostic Model
- [`xb_dia()`](https://xiaojie0519.github.io/E2E/reference/xb_dia.md) :
  Train an XGBoost Tree Model for Classification
- [`min_max_normalize()`](https://xiaojie0519.github.io/E2E/reference/min_max_normalize.md)
  : Min-Max Normalization
- [`Surv`](https://xiaojie0519.github.io/E2E/reference/Surv.md) :
  re-export Surv from survival
