# E2E: An R Package makes it Easy to Ensemble

Welcome to E2E, an R package designed to simplify ensemble modeling for both diagnostic and prognostic tasks. It provides a streamlined workflow for training, evaluating, and applying various single models and ensemble techniques like Bagging, Voting, and Stacking.

**Author:** Shanjie Luan (ORCID: 0009-0002-8569-8526)

**Citation:** If you use E2E in your research, please cite it as:
"Shanjie Luan (2025). E2E: An R Package that makes it easy to ensemble. [https://github.com/XIAOJIE0519/E2E](https://github.com/XIAOJIE0519/E2E)"

**Note:** The article is in the process of being written/submitted and is undergoing review by Cran and further revisions. If you have any questions, please contact Luan20050519@163.com.

---

## Installation

**(Note: This package is still under active development. While you're welcome to try it out, I do not recommend using it for production environments yet.)**

You can install the package directly from GitHub using `devtools` or `remotes`:

```R
# install.packages("devtools") # if you don't have it
devtools::install_github("XIAOJIE0519/E2E")
```

or

```R
# install.packages("remotes") # if you don't have it
remotes::install_github("XIAOJIE0519/E2E")
```

After installation, load the package:

```R
library(E2E)
```

---

## Getting Started: Core Principles

The E2E package distinguishes between **Diagnostic Models** (for classification) and **Prognostic Models** (for survival analysis). Each category offers a suite of single models and ensemble methods.

Before running any models, you **must initialize** the respective modeling system to register default models.

To follow the examples, you'll need sample data files. There are four data frame for you to have a try: train_dia, test_dia, train_pro, test_pro. 

`train_dia` and `test_dia` are for diagnosis, with column names sample, outcome, variable 1, 2, 3. `train_pro` and `test_pro` are for prognosis, with column names sample, outcome, time, variable 1, 2, 3.


## 1. Diagnostic Models (Classification)

**Initialization:**

```R
initialize_modeling_system_dia()
```

### 1.1 ALL Models (Run Multiple Single Models)

Run a selection or all available diagnostic models and print their summaries:

```R
results_dia <- models_dia(train_dia, model = c("rf", "lasso", "xb"), tune = FALSE)
print_model_summary_dia("rf", results_dia$rf) # View specific model summary
```

To run all registered models:

```R
results_all_dia <- models_dia(train_dia, model = "all_dia", tune = FALSE)
for (m_name in names(results_all_dia)) print_model_summary_dia(m_name, results_all_dia[[m_name]])
```

### 1.2 Bagging

Train a Bagging ensemble (e.g., with XGBoost as base learners):

```R
bagging_xb_results <- bagging_dia(train_dia, base_model_name = "xb", n_estimators = 5)
print_model_summary_dia("Bagging (XGBoost)", bagging_xb_results)
```

### 1.3 Voting

Train a Voting ensemble (e.g., soft voting based on top 3 AUROC models):

```R
voting_soft_results <- voting_dia(results_all_models = results_all_dia,
                                  train_dia, type = "soft",
                                  weight_metric = "AUROC", top = 3)
print_model_summary_dia("Voting (Soft)", voting_soft_results)
```

### 1.4 Stacking

Train a Stacking ensemble (e.g., with Lasso meta-model on top 3 AUROC models):

```R
stacking_lasso_results <- stacking_dia(results_all_models = results_all_dia,
                                       train_dia, meta_model_name = "lasso", top = 3)
print_model_summary_dia("Stacking (Lasso)", stacking_lasso_results)
```

### 1.5 imbalance (Easyensemble)

Train a imbalance ensemble (e.g., with XGBoost as base learners):

```R
results_imbalance_dia <- imbalance_dia(train_dia, base_model_name = "xb", n_estimators = 10,
                                       tune_base_model = FALSE, threshold_choices = "f1")
print_model_summary_dia("Imbalance (XGBoost)", results_imbalance_dia)
```

### 1.6 apply (Apply to New Data)

Apply a trained diagnostic model (e.g., Bagging) to new, unseen data:

```R
bagging_pred_new <- apply_dia(trained_model_object = bagging_xb_results$model_object,
                                                test_dia, pos_class = "Positive", neg_class = "Negative")
head(bagging_pred_new)
```

### 1.7 Figure (Visualization)

Generate ROC, PRC curves, and Confusion Matrix plots for diagnostic models:

```R
# For plotting, you often want evaluation results on a test set
# However, we here only provide a train set visualization

figure_dia(type = "roc", data = results_imbalance_dia, output_file = "Diagnostic_ROC", output_type = "pdf")
figure_dia(type = "prc", data = results_imbalance_dia, output_file = "Diagnostic_PRC", output_type = "png")
figure_dia(type = "matrix", data = results_imbalance_dia, output_file = "Diagnostic_Matrix", output_type = "pdf")
```

### 1.8 Add new models

Add Adaboost as an example:

```R
ab_dia <- function(X, y, tune = FALSE, cv_folds = 5, cv_repeats = 3) {
  set.seed(123)
  ctrl <- caret::trainControl(method = "repeatedcv", number = cv_folds, repeats = cv_repeats,
                              classProbs = TRUE, summaryFunction = twoClassSummary,
                              savePredictions = "final")
  if (tune) {
    grid <- expand.grid(iter = c(10, 20, 30, 40), maxdepth = c(1, 2), nu = 0.1)
  } else {
    grid <- expand.grid(iter = 20, maxdepth = 1, nu = 0.1) }
  model <- caret::train(x = X, y = y, method = "ada", metric = "ROC", trControl = ctrl, tuneGrid = grid)
  return(model)
}

register_model_dia("ab", ab_dia)
```

---

## 2. Prognostic Models (Survival Analysis)

**Initialization:**

```R
initialize_modeling_system_pro()
```

### 2.1 ALL Models (Run Multiple Single Models)

Run a selection or all available prognostic models and print their summaries:

```R
results_pro <- models_pro(train_pro, model = c("lasso_pro", "rsf_pro"))
print_model_summary_pro("lasso_pro", results_pro$lasso_pro)
```

To run all registered models:

```R
results_all_pro <- models_pro(train_pro, model = "all_pro")
for (m_name in names(results_all_pro)) print_model_summary_pro(m_name, results_all_pro[[m_name]])
```

### 2.2 Bagging

Train a Bagging ensemble (e.g., with GBM as base learners):

```R
bagging_gbm_pro_results <- bagging_pro(train_pro,
                                       base_model_name = "gbm_pro", n_estimators = 5)
print_model_summary_pro("Bagging (GBM)", bagging_gbm_pro_results)
```

### 2.3 Stacking

Train a Stacking ensemble (e.g., with GBM meta-model on top 3 C-index models):

```R
stacking_gbm_pro_results <- stacking_pro(results_all_models = results_all_pro,
                                         train_pro,
                                         meta_model_name = "gbm_pro", top = 3)
print_model_summary_pro("Stacking (GBM)", stacking_gbm_pro_results)
```

### 2.4 apply (Apply to New Data)

Apply a trained prognostic model (e.g., Bagging) to new, unseen data:

```R
bagging_pro_pred_new <- apply_pro(trained_model_object = bagging_gbm_pro_results$model_object, test_pro)
head(bagging_pro_pred_new)
```

### 2.5 Figure (Visualization)

Generate Kaplan-Meier (KM) and Time-Dependent ROC (tdROC) curves for prognostic models:

```R
# For plotting, you often want evaluation results on a test set
# However, we here only provide a train set visualization

figure_pro(type = "km", data = stacking_gbm_pro_results, output_file = "Prognostic_KM",
           output_type = "pdf", time_unit = "days")
figure_pro(type = "tdroc", data = stacking_gbm_pro_results, output_file = "Prognostic_TDROC",
           output_type = "png", time_unit = "days")
```

---

## 3. SHAP Explanation (for both Diagnostic & Prognostic)

Generate SHAP explanation plots (e.g., using XGBoost as a surrogate model) for a model's predictions.

```R
# Diagnostic Model SHAP (using Bagging results on train.csv as example)
figure_shap(data= bagging_xb_results, raw_data = train_dia,
            output_file = "Dia_SHAP_Example", model_type = "xgboost",
            output_type = "pdf", target_type = "diagnosis") 

# Prognostic Model SHAP (using Bagging results on train_prognosis.csv as example)
figure_shap(data = stacking_gbm_pro_results, raw_data = train_pro,
            output_file = "Pro_SHAP_Example", model_type = "xgboost",
            output_type = "pdf", target_type = "prognosis")
```

---

## 4. Principle and drawing results

### Principle

![q](https://github.com/user-attachments/assets/6a908218-f84d-4b40-83ed-a6c6acb0fe37)

### SHAP

![image](https://github.com/user-attachments/assets/aac6754d-c34b-49d1-987c-49e4b6417268)

### Other

![image](https://github.com/user-attachments/assets/59589d68-cd46-4b31-be84-839105121919)

![image](https://github.com/user-attachments/assets/f9adeea4-8635-44a2-b573-7017b8ca14ca)

---

