
---

# E2E: An R Package for Easy-to-Build Ensemble Models

**E2E** is a comprehensive R package designed to streamline the development, evaluation, and interpretation of machine learning models for both **diagnostic (classification)** and **prognostic (survival analysis)** tasks. It provides a robust, extensible framework for training individual models and building powerful ensembles—including Bagging, Voting, and Stacking—with minimal code. The package also includes integrated tools for visualization and model explanation via SHAP values.

**Author:** Shanjie Luan (ORCID: 0009-0002-8569-8526)

**Citation:** If you use E2E in your research, please cite it as:
"Shanjie Luan (2025). E2E: An R Package that makes it easy to ensemble. [https://github.com/XIAOJIE0519/E2E](https://github.com/XIAOJIE0519/E2E)"

**Note:** The article is in the process of being written/submitted and is undergoing review by CRAN and further revisions. If you have any questions, please contact [Luan20050519@163.com](mailto:Luan20050519@163.com).

---

## Installation

The development version of E2E can be installed directly from GitHub using `devtools` or `remotes`.

```R
# If you don't have devtools, install it first:
# install.packages("devtools")
devtools::install_github("XIAOJIE0519/E2E")
```

or

```R
# If you don't have remotes, install it first:
# install.packages("remotes")
remotes::install_github("XIAOJIE0519/E2E")
```

After installation, load the package into your R session:

```R
library(E2E)
```

---

## Core Concepts & Getting Started

E2E operates on two parallel tracks: **Diagnostic Models** and **Prognostic Models**. Before using functions from either track, you **must initialize** the corresponding system. This step registers a suite of pre-defined, commonly used models.
To follow the examples, you'll need sample data files. There are four data frame for you to have a try: train_dia, test_dia, train_pro, test_pro. 
train_dia and test_dia are for diagnosis, with column names sample, outcome, variable 1, 2, 3. train_pro and test_pro are for prognosis, with column names sample, outcome, time, variable 1, 2, 3.

---

## 1. Diagnostic Models (Classification)

This track is dedicated to binary classification tasks.

### 1.1 Initialization

First, initialize the diagnostic modeling system.

```R
initialize_modeling_system_dia()
```

### 1.2 Training Single Models with `models_dia`

The `models_dia` function is the gateway to training one or more standard classification models.

#### Basic Usage

By default, `models_dia` runs all registered models with standard settings.

```R
# Run all available diagnostic models
results_all_dia <- models_dia(train_dia)

# Print a summary for a specific model (e.g., Random Forest)
print_model_summary_dia("rf", results_all_dia$rf)
```

#### Advanced Usage & Customization

You can precisely control the modeling process by specifying parameters.

```R
# Run a specific subset of models with tuning enabled and custom thresholds
results_dia_custom <- models_dia(
  data = train_dia,
  model = c("rf", "lasso", "xb"),
  tune = TRUE,
  seed = 123,
  threshold_choices = list(rf = "f1", lasso = 0.6, xb = "youden"),
  positive_label_value = 1,
  negative_label_value = 0,
  new_positive_label = "Case",
  new_negative_label = "Control"
)

# View the custom results
print_model_summary_dia("rf", results_dia_custom$rf)
```

* **`data`**: Your training data frame. Must have ID in the 1st column, label in the 2nd, and features thereafter.
* **`model`**: A character vector of models to run (e.g., `c("rf", "lasso")`) or `"all_dia"` for all registered models.
* **`tune`**: A logical (`TRUE`/`FALSE`) to enable/disable hyperparameter tuning for the models. Default is `FALSE`.
* **`seed`**: An integer for ensuring reproducibility.
* **`threshold_choices`**: Strategy for calculating classification metrics.

  * `"default"`: Uses a 0.5 probability threshold.
  * `"f1"`: Finds the threshold that maximizes the F1-score.
  * `"youden"`: Finds the threshold that maximizes Youden's J statistic.
  * A numeric value (e.g., `0.6`): Applies a specific threshold to all models.
  * A named list (e.g., `list(rf = "f1", lasso = 0.6)`): Assigns different strategies to different models.
* **`positive_label_value` / `negative_label_value`**: The values in your `outcome` column that represent the positive (e.g., 1) and negative (e.g., 0) classes.
* **`new_positive_label` / `new_negative_label`**: The desired factor level names for the positive and negative classes (e.g., "Positive", "Negative").

### 1.3 Ensemble Modeling

#### Bagging (`bagging_dia`)

Builds a Bagging ensemble by training a base model on multiple bootstrap samples.

**Basic Usage**

```R
# Create a Bagging ensemble with XGBoost as the base model
bagging_xb_results <- bagging_dia(train_dia, base_model_name = "xb")
print_model_summary_dia("Bagging (XGBoost)", bagging_xb_results)
```

**Advanced Usage**

* **`base_model_name`**: The name of the registered model to use as the base estimator (e.g., `"rf"`, `"xb"`).
* **`n_estimators`**: The number of base models to train. Default is `50`.
* **`subset_fraction`**: The fraction of samples to draw for each bootstrap. Default is `0.632`.
* **`tune_base_model`**: A logical (`TRUE`/`FALSE`) to enable tuning for each base model.

#### Voting (`voting_dia`)

Combines predictions from multiple pre-trained models. Requires the output from `models_dia`.

**Basic Usage**

```R
# Create a soft voting ensemble from the top 3 models (ranked by AUROC)
voting_soft_results <- voting_dia(
  results_all_models = results_all_dia,
  data = train_dia,
  type = "soft"
)
print_model_summary_dia("Voting (Soft)", voting_soft_results)
```

**Advanced Usage**

* **`results_all_models`**: The list object returned by `models_dia`.
* **`data`**: The training data, used for evaluation.
* **`type`**: `"soft"` for weighted average of probabilities (recommended), or `"hard"` for majority class voting.
* **`weight_metric`**: For `type = "soft"`, the metric used to weight models (e.g., `"AUROC"`, `"F1"`). Default is `"AUROC"`.
* **`top`**: The number of top-performing models to include in the ensemble. Default is `5`.

#### Stacking (`stacking_dia`)

Uses predictions from base models as features to train a final meta-model. Requires the output from `models_dia`.

**Basic Usage**

```R
# Create a Stacking ensemble with Lasso as the meta-model
stacking_lasso_results <- stacking_dia(
  results_all_models = results_all_dia,
  data = train_dia,
  meta_model_name = "lasso"
)
print_model_summary_dia("Stacking (Lasso)", stacking_lasso_results)
```

**Advanced Usage**

* **`meta_model_name`**: The name of the registered model to use as the meta-model.
* **`top`**: The number of top base models (ranked by AUROC) whose predictions will be used as features. Default is `5`.
* **`tune_meta`**: A logical (`TRUE`/`FALSE`) to enable tuning for the meta-model.

#### Handling Imbalanced Data (`imbalance_dia`)

Implements the EasyEnsemble algorithm, which trains models on balanced subsets created by undersampling the majority class.

**Basic Usage**

```R
# Create an EasyEnsemble with XGBoost as the base model
results_imbalance_dia <- imbalance_dia(train_dia, base_model_name = "xb")
print_model_summary_dia("Imbalance (XGBoost)", results_imbalance_dia)
```

**Advanced Usage**

* **`base_model_name`**: The base estimator to train on each balanced subset.
* **`n_estimators`**: The number of subsets to create and models to train. Default is `10`.
* **`tune_base_model`**: A logical (`TRUE`/`FALSE`) to enable tuning for each base model.

### 1.4 Applying Models to New Data (`apply_dia`)

Use a trained model object to make predictions on a new, unseen dataset.

```R
# Apply the trained Bagging model to the test set
bagging_pred_new <- apply_dia(
  trained_model_object = bagging_xb_results$model_object,
  new_data = test_dia,
  label_col_name = "outcome",
  pos_class = "Case", # Must match the label used during training
  neg_class = "Control" # Must match the label used during training
)

# You can then evaluate these new predictions
eval_results_new <- evaluate_model_dia(
  precomputed_prob = bagging_pred_new$score,
  y_data = factor(test_dia$outcome, levels = c(0, 1), labels = c("Control", "Case")),
  sample_ids = test_dia$sample,
  pos_class = "Case",
  neg_class = "Control"
)
print(eval_results_new$evaluation_metrics)
```

### 1.5 Visualization (`figure_dia`)

Generate high-quality plots to evaluate model performance.

```R
# ROC Curve
figure_dia(type = "roc", data = results_imbalance_dia, output_file = "Diagnostic_ROC", output_type = "pdf")

# Precision-Recall Curve
figure_dia(type = "prc", data = results_imbalance_dia, output_file = "Diagnostic_PRC", output_type = "png")

# Confusion Matrix
figure_dia(type = "matrix", data = results_imbalance_dia, output_file = "Diagnostic_Matrix", output_type = "svg")
```

* **`type`**: The type of plot: `"roc"`, `"prc"`, or `"matrix"`.
* **`data`**: A results object from a model run (e.g., `bagging_dia`, `models_dia`).
* **`output_file`**: The base name for the saved file.
* **`output_type`**: The file format: `"pdf"`, `"png"`, or `"svg"`.

### 1.6 Extending the Framework: Adding New Models

The E2E framework is fully extensible. You can register your own custom models. For example, to add Adaboost:

```R
# 1. Define the model function (must accept X, y, and other standard args)
ab_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method = "cv", number = cv_folds,
                              classProbs = TRUE, summaryFunction = caret::twoClassSummary)
  grid <- if (tune) {
    expand.grid(iter = c(50, 100), maxdepth = c(1, 2), nu = 0.1)
  } else {
    expand.grid(iter = 50, maxdepth = 1, nu = 0.1)
  }
  caret::train(x = X, y = y, method = "ada", metric = "ROC", trControl = ctrl, tuneGrid = grid)
}

# 2. Register the model with a unique name
register_model_dia("ab", ab_dia)

# 3. Now you can use "ab" in any diagnostic function
results_ab <- models_dia(train_dia, model = "ab")
print_model_summary_dia("ab", results_ab$ab)
```

---

## 2. Prognostic Models (Survival Analysis)

This track is dedicated to survival prediction tasks.

### 2.1 Initialization

First, initialize the prognostic modeling system.

```R
initialize_modeling_system_pro()
```

### 2.2 Training Single Models with `models_pro`

The `models_pro` function trains one or more standard survival models.

#### Basic Usage

```R
# Run all available prognostic models
results_all_pro <- models_pro(train_pro)

# Print summary for Random Survival Forest
print_model_summary_pro("rsf_pro", results_all_pro$rsf_pro)
```

#### Advanced Usage & Customization

```R
# Run a subset of models with specific evaluation settings
results_pro_custom <- models_pro(
  data = train_pro,
  model = c("lasso_pro", "rsf_pro"),
  tune = TRUE,
  seed = 123,
  time_unit = "day",
  years_to_evaluate = c(1, 3, 5)
)
```

* **`data`**: Your training data frame. Must have ID in the 1st column, outcome status (0/1) in the 2nd, time in the 3rd, and features thereafter.
* **`model`**: A character vector of models (e.g., `c("lasso_pro", "rsf_pro")`) or `"all_pro"`.
* **`tune`**: A logical (`TRUE`/`FALSE`) for hyperparameter tuning.
* **`time_unit`**: The unit of time in your `time` column: `"day"`, `"month"`, or `"year"`. All times are converted to days internally for consistency. Default is `"day"`.
* **`years_to_evaluate`**: A numeric vector of years for calculating time-dependent AUROC. Default is `c(1, 3, 5)`.

### 2.3 Ensemble Modeling

#### Bagging (`bagging_pro`)

Builds a Bagging ensemble for survival models.

**Basic Usage**

```R
# Create a Bagging ensemble with GBM as the base survival model
bagging_gbm_pro_results <- bagging_pro(train_pro, base_model_name = "gbm_pro")
print_model_summary_pro("Bagging (GBM)", bagging_gbm_pro_results)
```

**Advanced Usage**

* **`base_model_name`**: The registered survival model to use as the base estimator.
* **`n_estimators`**: The number of base models. Default is `10`.

#### Stacking (`stacking_pro`)

Builds a Stacking ensemble for survival models.

**Basic Usage**

```R
# Create a Stacking ensemble with GBM as the meta-model
# It uses the top models from `results_all_pro` ranked by C-index
stacking_gbm_pro_results <- stacking_pro(
  results_all_models = results_all_pro,
  data = train_pro,
  meta_model_name = "gbm_pro"
)
print_model_summary_pro("Stacking (GBM)", stacking_gbm_pro_results)
```

**Advanced Usage**

* **`meta_model_name`**: The registered survival model to use as the meta-model.
* **`top`**: The number of top base models (ranked by C-index) to use. Default is `3`.

### 2.4 Applying Models to New Data (`apply_pro`)

Generate prognostic scores for a new dataset using a trained survival model.

```R
# Apply the trained stacking model to the test set
pro_pred_new <- apply_pro(
  trained_model_object = stacking_gbm_pro_results$model_object,
  new_data = test_pro,
  time_unit = "day"
)
head(pro_pred_new)

# Evaluate the new prognostic scores
eval_pro_new <- evaluate_model_pro(
  precomputed_score = pro_pred_new$score,
  Y_surv_obj = survival::Surv(time = test_pro$time, event = test_pro$outcome),
  sample_ids = test_pro$sample,
  years_to_evaluate = c(1, 3, 5)
)
print(eval_pro_new$evaluation_metrics)
```

### 2.5 Visualization (`figure_pro`)

Generate Kaplan-Meier (KM) and time-dependent ROC (tdROC) curves.

```R
# Kaplan-Meier Curve (groups based on median risk score)
figure_pro(type = "km", data = stacking_gbm_pro_results, output_file = "Prognostic_KM",
           output_type = "pdf", time_unit = "days")

# Time-Dependent ROC Curve
figure_pro(type = "tdroc", data = stacking_gbm_pro_results, output_file = "Prognostic_TDROC",
           output_type = "png", time_unit = "days")
```

* **`type`**: The type of plot: `"km"` or `"tdroc"`.
* **`data`**: A results object from a prognostic model run.
* **`time_unit`**: The time unit of the data, used for labeling axes correctly.

---

## 3. Model Explanation with SHAP (`figure_shap`)

The `figure_shap` function provides model-agnostic explanations by training a simpler surrogate model (e.g., XGBoost) on the predictions of your complex model and calculating SHAP values. This reveals which features had the most impact on the model's output.

### Basic Usage

```R
# Explain a diagnostic model's predictions (e.g., from Bagging)
figure_shap(
  data = bagging_xb_results,
  raw_data = train_dia,
  output_file = "Dia_SHAP_Example",
  target_type = "diagnosis"
)

# Explain a prognostic model's predictions (e.g., from Stacking)
figure_shap(
  data = stacking_gbm_pro_results,
  raw_data = train_pro,
  output_file = "Pro_SHAP_Example",
  target_type = "prognosis"
)
```

### Parameters

* **`data`**: The results object from any E2E model run. It must contain the `sample_score` data frame with model predictions.
* **`raw_data`**: The original data frame containing the features. This is crucial for linking predictions back to input features.
* **`output_file`**: The base name for the saved plot.
* **`model_type`**: The surrogate model to use for SHAP calculation: `"xgboost"` (default) or `"lasso"`.
* **`output_type`**: The file format: `"pdf"`, `"png"`, or `"svg"`.
* **`target_type`**: `"diagnosis"` or `"prognosis"`. This is **critical** as it tells the function how to parse the `raw_data` to correctly identify the feature columns.

---

## 4. Principle and Example Results

### Methodological Framework

![Workflow](https://github.com/user-attachments/assets/6a908218-f84d-4b40-83ed-a6c6acb0fe37)

### SHAP Explanation Plot

![SHAP Example](https://github.com/user-attachments/assets/fd9f21c3-8104-4dce-a449-62d59d191cc4)

### Other Visualization Examples

![Confusion Matrix](https://github.com/user-attachments/assets/3f89f1c7-3e27-4ace-a331-f5a6b040df15)
![ROC Curve](https://github.com/user-attachments/assets/a7a002e0-8f65-4c34-bc73-07db57f9a2c0)
![Kaplan-Meier Curve](https://github.com/user-attachments/assets/d21df09c-9492-42a9-96fd-46385bcab48d)
![Time-Dependent ROC](https://github.com/user-attachments/assets/971c3a7f-3927-4fce-baeb-33401f9159ad)
