# diagnosis.R
#' @importFrom utils globalVariables
utils::globalVariables(c("x", "y", "recall", "Actual", "Predicted", "Freq", "Percentage",
                         "time", "AUROC", "feature", "value", "ID", "e",
                         "score_col", "label", "sample", "score", ".", "precision", "Label"))

# Internal package environment for model registry.
# This environment holds functions for various diagnostic models, allowing them
# to be registered and retrieved dynamically. It also stores global settings
# like positive/negative label values.
.model_registry_env_dia <- new.env()
.model_registry_env_dia$known_models_internal <- list()
.model_registry_env_dia$is_initialized <- FALSE
.model_registry_env_dia$pos_label_value <- 1 # Default positive label numeric value
.model_registry_env_dia$neg_label_value <- 0 # Default negative label numeric value

# List of required packages for all diagnostic functions.
# This list is used by initialize_modeling_system_dia to check for dependencies.
required_packages_dia <- c(
  "readr", "dplyr", "caret", "pROC", "PRROC",
  "randomForest", "xgboost", "e1071", "nnet",
  "glmnet", "MASS", "klaR", "gbm", "rpart"
)


#' @title Prepare Data for Diagnostic Models (Internal)
#' @description Prepares an input data frame by separating ID, label, and features
#'   based on a fixed column structure (1st=ID, 2nd=Label, 3rd+=Features).
#'
#' @param data A data frame where the first column is the sample ID, the second
#'   is the outcome label, and subsequent columns are features.
#' @param positive_label_value A numeric or character value that represents
#'   the positive class in the raw data.
#' @param negative_label_value A numeric or character value that represents
#'   the negative class in the raw data.
#' @param new_positive_label A character string, the desired factor level name
#'   for the positive class (e.g., "Positive").
#' @param new_negative_label A character string, the desired factor level name
#'   for the negative class (e.g., "Negative").
#' @return A list containing `X`, `y`, `sample_ids`, class labels, and original y.
#' @noRd
.prepare_data_dia <- function(data,
                              positive_label_value = 1, negative_label_value = 0,
                              new_positive_label = "Positive", new_negative_label = "Negative") {
  if (!is.data.frame(data)) {
    stop("Input 'data' must be a data frame.")
  }
  if (ncol(data) < 3) {
    stop("Input data must have at least three columns: ID, Label, and at least one Feature.")
  }

  sample_ids <- data[[1]]
  y_original_numeric <- data[[2]]
  X <- data[, -c(1, 2), drop = FALSE]

  # Convert label to factor
  y <- base::factor(y_original_numeric,
                    levels = c(negative_label_value, positive_label_value),
                    labels = c(new_negative_label, new_positive_label))

  # Ensure feature columns are of appropriate types (factor or numeric)
  for (col_name in names(X)) {
    if (is.character(X[[col_name]])) {
      X[[col_name]] <- base::as.factor(X[[col_name]])
    } else if (!is.numeric(X[[col_name]]) && !base::is.factor(X[[col_name]])) {
      # A simple heuristic to decide if a non-numeric column should be a factor
      if (all(is.na(suppressWarnings(as.numeric(as.character(X[[col_name]]))))) && !all(is.na(X[[col_name]]))) {
        X[[col_name]] <- base::as.factor(X[[col_name]])
      } else {
        X[[col_name]] <- as.numeric(as.character(X[[col_name]]))
      }
    }
  }

  list(
    X = as.data.frame(X),
    y = y,
    sample_ids = sample_ids,
    pos_class_label = new_positive_label,
    neg_class_label = new_negative_label,
    y_original_numeric = y_original_numeric
  )
}

# ------------------------------------------------------------------------------
# Model Registry and Utility Functions
# ------------------------------------------------------------------------------

#' @title Register a Diagnostic Model Function
#' @description Registers a user-defined or pre-defined diagnostic model function
#'   with the internal model registry. This allows the function to be called
#'   later by its registered name, facilitating a modular model management system.
#'
#' @param name A character string, the unique name to register the model under.
#' @param func A function, the R function implementing the diagnostic model.
#'   This function should typically accept `X` (features) and `y` (labels)
#'   as its first two arguments and return a `caret::train` object.
#' @return NULL. The function registers the model function invisibly.
#' @examples
#' \donttest{
#' # Example of a dummy model function for registration
#' my_dummy_rf_model <- function(X, y, tune = FALSE, cv_folds = 5) {
#'   message("Training dummy RF model...")
#'   # This is a placeholder and doesn't train a real model.
#'   # It returns a list with a structure similar to a caret train object.
#'   list(method = "dummy_rf")
#' }
#'
#' # Initialize the system before registering
#' initialize_modeling_system_dia()
#'
#' # Register the new model
#' register_model_dia("dummy_rf", my_dummy_rf_model)
#'
#' # Verify that the model is now in the list of registered models
#' "dummy_rf" %in% names(get_registered_models_dia())
#' }
#' @seealso \code{\link{get_registered_models_dia}}, \code{\link{initialize_modeling_system_dia}}
#' @export
register_model_dia <- function(name, func) {
  if (!is.character(name) || length(name) != 1 || nchar(name) == 0) {
    stop("Model name must be a non-empty character string.")
  }
  if (!is.function(func)) {
    stop("Model function must be an R function.")
  }
  .model_registry_env_dia$known_models_internal[[name]] <- func
}

#' @title Get Registered Diagnostic Models
#' @description Retrieves a list of all diagnostic model functions currently
#'   registered in the internal environment.
#'
#' @return A named list where names are the registered model names and values
#'   are the corresponding model functions.
#' @examples
#' \donttest{
#' # Ensure system is initialized to see the default models
#' initialize_modeling_system_dia()
#' models <- get_registered_models_dia()
#' # See available model names
#' print(names(models))
#' }
#' @seealso \code{\link{register_model_dia}}, \code{\link{initialize_modeling_system_dia}}
#' @export
get_registered_models_dia <- function() {
  return(.model_registry_env_dia$known_models_internal)
}

#' @title Calculate Classification Metrics at a Specific Threshold
#' @description Calculates various classification performance metrics (Accuracy,
#'   Precision, Recall, F1-score, Specificity, True Positives, etc.) for binary
#'   classification at a given probability threshold.
#'
#' @param prob_positive A numeric vector of predicted probabilities for the
#'   positive class.
#' @param y_true A factor vector of true class labels.
#' @param threshold A numeric value between 0 and 1, the probability threshold
#'   above which a prediction is considered positive.
#' @param pos_class A character string, the label for the positive class.
#' @param neg_class A character string, the label for the negative class.
#' @return A list containing:
#'   \itemize{
#'     \item `Threshold`: The threshold used.
#'     \item `Accuracy`: Overall prediction accuracy.
#'     \item `Precision`: Precision for the positive class.
#'     \item `Recall`: Recall (Sensitivity) for the positive class.
#'     \item `F1`: F1-score for the positive class.
#'     \item `Specificity`: Specificity for the negative class.
#'     \item `TP`, `TN`, `FP`, `FN`, `N`: Counts of True Positives, True Negatives,
#'       False Positives, False Negatives, and total samples.
#'   }
#' @examples
#' y_true_ex <- factor(c("Negative", "Positive", "Positive", "Negative", "Positive"),
#'                     levels = c("Negative", "Positive"))
#' prob_ex <- c(0.1, 0.8, 0.6, 0.3, 0.9)
#' metrics <- calculate_metrics_at_threshold_dia(
#'   prob_positive = prob_ex,
#'   y_true = y_true_ex,
#'   threshold = 0.5,
#'   pos_class = "Positive",
#'   neg_class = "Negative"
#' )
#' print(metrics)
#' @importFrom caret confusionMatrix
#' @export
calculate_metrics_at_threshold_dia <- function(prob_positive, y_true, threshold, pos_class, neg_class) {
  y_true <- factor(y_true, levels = c(neg_class, pos_class))
  y_pred_class <- factor(base::ifelse(prob_positive >= threshold, pos_class, neg_class),
                         levels = c(neg_class, pos_class))
  cm_obj <- caret::confusionMatrix(y_pred_class, y_true, positive = pos_class)

  TP <- cm_obj$table[pos_class, pos_class]
  TN <- cm_obj$table[neg_class, neg_class]
  FP <- cm_obj$table[pos_class, neg_class]
  FN <- cm_obj$table[neg_class, pos_class]
  N <- sum(cm_obj$table)

  metrics <- suppressWarnings(cm_obj$byClass)
  overall_metrics <- suppressWarnings(cm_obj$overall)

  f1 <- metrics["F1"]
  accuracy <- overall_metrics["Accuracy"]
  precision <- metrics["Precision"]
  recall <- metrics["Recall"]
  specificity <- metrics["Specificity"]

  return(list(
    Threshold = threshold,
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1 = f1,
    Specificity = specificity,
    TP = TP, TN = TN, FP = FP, FN = FN, N = N
  ))
}

#' @title Find Optimal Probability Threshold
#' @description Determines an optimal probability threshold for binary
#'   classification based on maximizing F1-score or Youden's J statistic.
#'
#' @param prob_positive A numeric vector of predicted probabilities for the
#'   positive class.
#' @param y_true A factor vector of true class labels.
#' @param type A character string, specifying the optimization criterion:
#'   "f1" for F1-score or "youden" for Youden's J statistic (Sensitivity + Specificity - 1).
#' @param pos_class A character string, the label for the positive class.
#' @param neg_class A character string, the label for the negative class.
#' @return A numeric value, the optimal probability threshold.
#' @examples
#' y_true_ex <- factor(c("Negative", "Positive", "Positive", "Negative", "Positive"),
#'                     levels = c("Negative", "Positive"))
#' prob_ex <- c(0.1, 0.8, 0.6, 0.3, 0.9)
#'
#' # Find threshold maximizing F1-score
#' opt_f1_threshold <- find_optimal_threshold_dia(
#'   prob_positive = prob_ex,
#'   y_true = y_true_ex,
#'   type = "f1",
#'   pos_class = "Positive",
#'   neg_class = "Negative"
#' )
#' print(opt_f1_threshold)
#'
#' # Find threshold maximizing Youden's J
#' opt_youden_threshold <- find_optimal_threshold_dia(
#'   prob_positive = prob_ex,
#'   y_true = y_true_ex,
#'   type = "youden",
#'   pos_class = "Positive",
#'   neg_class = "Negative"
#' )
#' print(opt_youden_threshold)
#' @importFrom caret confusionMatrix
#' @export
find_optimal_threshold_dia <- function(prob_positive, y_true, type = c("f1", "youden"), pos_class, neg_class) {
  type <- match.arg(type)
  thresholds <- unique(sort(c(0, prob_positive, 1)))
  thresholds <- thresholds[thresholds > 0 & thresholds < 1]
  if (length(thresholds) == 0) thresholds <- 0.5 # Fallback if no unique internal probabilities

  best_score <- -Inf
  best_threshold <- 0.5
  y_true_factor <- factor(y_true, levels = c(neg_class, pos_class))

  for (t in thresholds) {
    y_pred_class <- factor(base::ifelse(prob_positive >= t, pos_class, neg_class),
                           levels = c(neg_class, pos_class))
    cm <- suppressWarnings(caret::confusionMatrix(y_pred_class, y_true_factor, positive = pos_class))
    current_score <- NA
    if (type == "f1") {
      current_score <- cm$byClass["F1"]
    } else if (type == "youden") {
      current_score <- cm$byClass["Sensitivity"] + cm$byClass["Specificity"] - 1
    }
    if (!is.na(current_score) && current_score > best_score) {
      best_score <- current_score
      best_threshold <- t
    }
  }
  return(best_threshold)
}

#' @title Load and Prepare Data for Diagnostic Models
#' @description Loads a CSV file containing patient data, extracts features,
#'   and converts the label column into a factor suitable for classification
#'   models. Handles basic data cleaning like trimming whitespace and type conversion.
#'
#' @param data_path A character string, the file path to the input CSV data.
#'   The first column is assumed to be a sample ID.
#' @param label_col_name A character string, the name of the column containing
#'   the class labels.
#' @param positive_label_value A numeric or character value that represents
#'   the positive class in the raw data.
#' @param negative_label_value A numeric or character value that represents
#'   the negative class in the raw data.
#' @param new_positive_label A character string, the desired factor level name
#'   for the positive class (e.g., "Positive").
#' @param new_negative_label A character string, the desired factor level name
#'   for the negative class (e.g., "Negative").
#'
#' @return A list containing:
#'   \itemize{
#'     \item `X`: A data frame of features (all columns except ID and label).
#'     \item `y`: A factor vector of class labels, with levels `new_negative_label`
#'       and `new_positive_label`.
#'     \item `sample_ids`: A vector of sample IDs (the first column of the input data).
#'     \item `pos_class_label`: The character string used for the positive class factor level.
#'     \item `neg_class_label`: The character string used for the negative class factor level.
#'     \item `y_original_numeric`: The original numeric/character vector of labels.
#'   }
#' @examples
#' \donttest{
#' # Create a dummy CSV file in a temporary directory for demonstration
#' temp_csv_path <- tempfile(fileext = ".csv")
#' dummy_data <- data.frame(
#'   ID = paste0("Patient", 1:50),
#'   Disease_Status = sample(c(0, 1), 50, replace = TRUE),
#'   FeatureA = rnorm(50),
#'   FeatureB = runif(50, 0, 100),
#'   CategoricalFeature = sample(c("X", "Y", "Z"), 50, replace = TRUE)
#' )
#' write.csv(dummy_data, temp_csv_path, row.names = FALSE)
#'
#' # Load and prepare data from the temporary file
#' prepared_data <- load_and_prepare_data_dia(
#'   data_path = temp_csv_path,
#'   label_col_name = "Disease_Status",
#'   positive_label_value = 1,
#'   negative_label_value = 0,
#'   new_positive_label = "Case",
#'   new_negative_label = "Control"
#' )
#'
#' # Check prepared data structure
#' str(prepared_data$X)
#' table(prepared_data$y)
#'
#' # Clean up the dummy file
#' unlink(temp_csv_path)
#' }
#' @importFrom readr read_csv
#' @export
load_and_prepare_data_dia <- function(data_path, label_col_name,
                                      positive_label_value = 1, negative_label_value = 0,
                                      new_positive_label = "Positive", new_negative_label = "Negative") {
  df_original <- readr::read_csv(data_path, show_col_types = FALSE)
  names(df_original) <- trimws(names(df_original))

  if (ncol(df_original) < 2) {
    stop("Input data must have at least two columns: an ID column (first column) and a label column (plus features if any).")
  }

  sample_ids <- df_original[[1]]
  df_features_and_label <- df_original[, -1, drop = FALSE]

  if (!label_col_name %in% names(df_features_and_label)) {
    stop(paste("Error: Label column '", label_col_name, "' not found in data after removing the first column (ID).", sep=""))
  }

  y_original_numeric <- df_features_and_label[[label_col_name]]

  y <- base::factor(y_original_numeric,
                    levels = c(negative_label_value, positive_label_value),
                    labels = c(new_negative_label, new_positive_label))
  X <- df_features_and_label[, setdiff(names(df_features_and_label), label_col_name), drop = FALSE]

  for (col_name in names(X)) {
    if (is.character(X[[col_name]])) {
      X[[col_name]] <- base::as.factor(X[[col_name]])
    } else if (!is.numeric(X[[col_name]]) && !base::is.factor(X[[col_name]])) {
      if (all(is.na(as.numeric(X[[col_name]]))) && !all(is.na(X[[col_name]]))) {
        X[[col_name]] <- base::as.factor(X[[col_name]])
      } else {
        X[[col_name]] <- as.numeric(X[[col_name]])
      }
    }
  }

  list(
    X = as.data.frame(X),
    y = y,
    sample_ids = sample_ids,
    pos_class_label = new_positive_label,
    neg_class_label = new_negative_label,
    y_original_numeric = y_original_numeric
  )
}

# ------------------------------------------------------------------------------
# Base Diagnostic Model Training Functions
# ------------------------------------------------------------------------------

#' @title Train a Random Forest Model for Classification
#' @description Trains a Random Forest model using `caret::train` for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning using `caret`'s
#'   default grid (if `TRUE`) or use a fixed `mtry` value (if `FALSE`).
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained Random Forest model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' rf_model <- rf_dia(X_toy, y_toy)
#' print(rf_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
rf_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method="cv", number=cv_folds,
                              classProbs=TRUE, summaryFunction=caret::twoClassSummary)

  grid <- if (tune) {
    expand.grid(
      mtry = c(floor(sqrt(ncol(X))),
               floor(ncol(X)/3),
               floor(ncol(X)/2),
               max(1, floor(ncol(X)*0.7)))  # 增加特征采样选项
    )
  } else {
    data.frame(mtry = floor(sqrt(ncol(X))))
  }

  caret::train(x=X, y=y, method="rf", metric="ROC",
               trControl=ctrl, tuneGrid=grid,
               ntree=500,
               nodesize=if(tune) 5 else 1,
               maxnodes=if(tune) NULL else NULL,
               tuneLength=if(tune && is.null(grid)) 5 else 1)
}

#' @title Train an XGBoost Tree Model for Classification
#' @description Trains an Extreme Gradient Boosting (XGBoost) model using `caret::train`
#'   for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning using `caret`'s
#'   default grid (if `TRUE`) or use fixed values (if `FALSE`).
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @param tune_length An integer, the number of random parameter combinations to try
#'   when tune=TRUE. Only used when search="random". Default is 20.
#' @return A `caret::train` object representing the trained XGBoost model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' xb_model <- xb_dia(X_toy, y_toy)
#' print(xb_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
xb_dia <- function(X, y, tune = FALSE, cv_folds = 5, tune_length = 20) {

  if (tune) {
    ctrl <- caret::trainControl(
      method = "cv",
      number = cv_folds,
      classProbs = TRUE,
      summaryFunction = caret::twoClassSummary,
      search = "random",
      allowParallel = TRUE
    )

    grid <- expand.grid(
      nrounds = c(50, 100),
      max_depth = c(2, 3, 4, 6),
      eta = c(0.01, 0.05, 0.1, 0.3),
      gamma = c(0, 0.1, 0.5, 1, 2),
      colsample_bytree = c(0.5, 0.7, 0.9, 1),
      min_child_weight = c(1, 3, 5, 7),
      subsample = c(0.6, 0.8, 1)
    )

    model <- caret::train(
      x = X, y = y,
      method = "xgbTree",
      metric = "ROC",
      trControl = ctrl,
      tuneLength = tune_length
    )

  } else {
    ctrl <- caret::trainControl(
      method = "cv",
      number = cv_folds,
      classProbs = TRUE,
      summaryFunction = caret::twoClassSummary
    )

    grid <- expand.grid(
      nrounds = 100, max_depth = 3, eta = 0.3, gamma = 0,
      colsample_bytree = 1, min_child_weight = 1, subsample = 1
    )

    model <- caret::train(
      x = X, y = y,
      method = "xgbTree",
      metric = "ROC",
      trControl = ctrl,
      tuneGrid = grid
    )
  }

  return(model)
}

#' @title Train a Support Vector Machine (Linear Kernel) Model for Classification
#' @description Trains a Support Vector Machine (SVM) model with a linear kernel
#'   using `caret::train` for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning using `caret`'s
#'   default grid (if `TRUE`) or a fixed value (if `FALSE`).
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained SVM model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' svm_model <- svm_dia(X_toy, y_toy)
#' print(svm_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
svm_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method="cv", number=cv_folds,
                              classProbs=TRUE, summaryFunction=caret::twoClassSummary)

  grid <- if (tune) {
    expand.grid(
      C = c(0.01, 0.1, 1, 10, 100)
    )
  } else {
    expand.grid(C = 1)
  }

  caret::train(x=X, y=y, method="svmLinear", metric="ROC",
               trControl=ctrl, tuneGrid=grid)
}


#' @title Train a Multi-Layer Perceptron (Neural Network) Model for Classification
#' @description Trains a Multi-Layer Perceptron (MLP) neural network model
#'   using `caret::train` for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning using `caret`'s
#'   default grid (if `TRUE`) or a fixed value (if `FALSE`).
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained MLP model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' mlp_model <- mlp_dia(X_toy, y_toy)
#' print(mlp_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @importFrom RSNNS mlp
#' @export
mlp_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method="cv", number=cv_folds,
                              classProbs=TRUE, summaryFunction=caret::twoClassSummary)

  grid <- if (tune) {
    expand.grid(
      size = c(3, 5, 10, 15, 20)
    )
  } else {
    expand.grid(size = 5)
  }

  caret::train(x=X, y=y, method="mlp", metric="ROC",
               trControl=ctrl, tuneGrid=grid,
               maxit = 1000,
               learnFunc = "Std_Backpropagation",
               learnFuncParams = c(0.2, 0),
               linOut = FALSE)
}


#' @title Train a Lasso (L1 Regularized Logistic Regression) Model for Classification
#' @description Trains a Lasso-regularized logistic regression model using `caret::train`
#'   (via `glmnet` method) for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning for `lambda`
#'   (if `TRUE`) or use a fixed value (if `FALSE`). `alpha` is fixed at 1 for Lasso.
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained Lasso model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' lasso_model <- lasso_dia(X_toy, y_toy)
#' print(lasso_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
lasso_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method="cv", number=cv_folds,
                              classProbs=TRUE, summaryFunction=caret::twoClassSummary)

  grid <- expand.grid(
    alpha = 1,
    lambda = if (tune) {
      c(10^seq(-5, -1, length=15), seq(0.1, 1, length=10))
    } else {
      0.01
    }
  )

  caret::train(x=X, y=y, method="glmnet", metric="ROC",
               trControl=ctrl, tuneGrid=grid,
               standardize = TRUE,
               thresh = 1e-7)
}


#' @title Train an Elastic Net (L1 and L2 Regularized Logistic Regression) Model for Classification
#' @description Trains an Elastic Net-regularized logistic regression model
#'   using `caret::train` (via `glmnet` method) for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning for `lambda`
#'   (if `TRUE`) or use a fixed value (if `FALSE`). `alpha` is fixed at 0.5 for Elastic Net.
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained Elastic Net model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' en_model <- en_dia(X_toy, y_toy)
#' print(en_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
en_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method="cv", number=cv_folds,
                              classProbs=TRUE, summaryFunction=caret::twoClassSummary)

  grid <- expand.grid(
    alpha = 0.5,
    lambda = if (tune) {
      c(10^seq(-5, -1, length=15), seq(0.1, 1, length=10))
    } else {
      0.01
    }
  )

  caret::train(x=X, y=y, method="glmnet", metric="ROC",
               trControl=ctrl, tuneGrid=grid,
               standardize = TRUE,
               thresh = 1e-7)
}

#' @title Train a Ridge (L2 Regularized Logistic Regression) Model for Classification
#' @description Trains a Ridge-regularized logistic regression model using `caret::train`
#'   (via `glmnet` method) for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning for `lambda`
#'   (if `TRUE`) or use a fixed value (if `FALSE`). `alpha` is fixed at 0 for Ridge.
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained Ridge model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' ridge_model <- ridge_dia(X_toy, y_toy)
#' print(ridge_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
ridge_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method="cv", number=cv_folds,
                              classProbs=TRUE, summaryFunction=caret::twoClassSummary)

  grid <- expand.grid(
    alpha = 0,
    lambda = if (tune) {
      c(10^seq(-5, -1, length=15), seq(0.1, 1, length=10))
    } else {
      0.01
    }
  )

  caret::train(x=X, y=y, method="glmnet", metric="ROC",
               trControl=ctrl, tuneGrid=grid,
               standardize = TRUE,
               thresh = 1e-7)
}

#' @title Train a Linear Discriminant Analysis (LDA) Model for Classification
#' @description Trains a Linear Discriminant Analysis (LDA) model using `caret::train`
#'   for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning (currently ignored for LDA).
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained LDA model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' lda_model <- lda_dia(X_toy, y_toy)
#' print(lda_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
lda_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method = "cv", number = cv_folds,
                              classProbs = TRUE, summaryFunction = caret::twoClassSummary)
  caret::train(x = X, y = y, method = "lda", metric = "ROC", trControl = ctrl)
}

#' @title Train a Quadratic Discriminant Analysis (QDA) Model for Classification
#' @description Trains a Quadratic Discriminant Analysis (QDA) model using `caret::train`
#'   for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning (currently ignored for QDA).
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained QDA model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' qda_model <- qda_dia(X_toy, y_toy)
#' print(qda_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
qda_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method = "cv", number = cv_folds,
                              classProbs = TRUE, summaryFunction = caret::twoClassSummary)
  caret::train(x = X, y = y, method = "qda", metric = "ROC", trControl = ctrl)
}

#' @title Train a Naive Bayes Model for Classification
#' @description Trains a Naive Bayes model using `caret::train` for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning using `caret`'s
#'   default grid (if `TRUE`) or fixed values (if `FALSE`).
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained Naive Bayes model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' nb_model <- nb_dia(X_toy, y_toy)
#' print(nb_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
nb_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(method = "cv", number = cv_folds,
                              classProbs = TRUE, summaryFunction = caret::twoClassSummary)
  grid <- if (tune) expand.grid(usekernel = c(TRUE, FALSE), fL = c(0, 1), adjust = c(0, 1)) else expand.grid(usekernel = TRUE, fL = 0, adjust = 1)
  caret::train(x = X, y = y, method = "nb", metric = "ROC", trControl = ctrl, tuneGrid = grid)
}

#' @title Train a Decision Tree Model for Classification
#' @description Trains a single Decision Tree model using `caret::train` (via `rpart` method)
#'   for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning for `cp`
#'   (complexity parameter) (if `TRUE`) or use a fixed value (if `FALSE`).
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @return A `caret::train` object representing the trained Decision Tree model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model
#' dt_model <- dt_dia(X_toy, y_toy)
#' print(dt_model)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
dt_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(
    method = "cv",
    number = cv_folds,
    classProbs = TRUE,
    summaryFunction = caret::twoClassSummary
  )

  n_samples <- nrow(X)

  grid <- if (tune) {

    expand.grid(
      cp = c(0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
    )
  } else {
    expand.grid(cp = 0.01)
  }

  # 设置rpart控制参数以防止过拟合
  rpart_control <- rpart::rpart.control(
    minsplit = max(2, floor(n_samples * 0.05)),
    minbucket = max(1, floor(n_samples * 0.02)),
    maxdepth = if(tune) 30 else 10,
    xval = 0
  )

  model <- tryCatch({
    caret::train(
      x = X,
      y = y,
      method = "rpart",
      metric = "ROC",
      trControl = ctrl,
      tuneGrid = grid,
      control = rpart_control
    )
  }, error = function(e) {
    warning(paste("Decision tree training failed with error:", e$message,
                  "\nRetrying with default parameters..."))

    grid_fallback <- expand.grid(cp = 0.01)
    caret::train(
      x = X,
      y = y,
      method = "rpart",
      metric = "ROC",
      trControl = ctrl,
      tuneGrid = grid_fallback,
      control = rpart::rpart.control(
        minsplit = 20,
        minbucket = 7,
        maxdepth = 10
      )
    )
  })

  return(model)
}

#' @title Train a Gradient Boosting Machine (GBM) Model for Classification
#' @description Trains a Gradient Boosting Machine (GBM) model using `caret::train`
#'   for binary classification.
#'
#' @param X A data frame of features.
#' @param y A factor vector of class labels.
#' @param tune Logical, whether to perform hyperparameter tuning for `interaction.depth`,
#'   `n.trees`, and `shrinkage` (if `TRUE`) or use fixed values (if `FALSE`).
#' @param cv_folds An integer, the number of cross-validation folds for `caret`.
#' @param tune_length An integer, the number of random parameter combinations to try
#'   when tune=TRUE. Only used when search="random". Default is 20.
#' @return A `caret::train` object representing the trained GBM model.
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 200
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#'
#' # Train the model with default parameters
#' gbm_model <- gbm_dia(X_toy, y_toy)
#' print(gbm_model)
#'
#' # Train with extensive tuning (random search)
#' gbm_model_tuned <- gbm_dia(X_toy, y_toy, tune = TRUE, tune_length = 30)
#' print(gbm_model_tuned)
#' }
#' @importFrom caret train trainControl twoClassSummary
#' @export
gbm_dia <- function(X, y, tune = FALSE, cv_folds = 5, tune_length = 10) {

  if (tune) {
    ctrl <- caret::trainControl(
      method = "cv",
      number = cv_folds,
      classProbs = TRUE,
      summaryFunction = caret::twoClassSummary,
      search = "random",
      allowParallel = TRUE
    )

    grid <- expand.grid(
      interaction.depth = c(1, 2, 3, 5, 7, 10),
      n.trees = c(50, 100, 150, 200, 300, 500),
      shrinkage = c(0.001, 0.01, 0.05, 0.1, 0.2),
      n.minobsinnode = c(5, 10, 15, 20, 30)
    )

    model <- caret::train(
      x = X,
      y = y,
      method = "gbm",
      metric = "ROC",
      trControl = ctrl,
      verbose = FALSE,
      tuneLength = tune_length,
      bag.fraction = 0.5
    )

  } else {
    ctrl <- caret::trainControl(
      method = "cv",
      number = cv_folds,
      classProbs = TRUE,
      summaryFunction = caret::twoClassSummary
    )

    grid <- expand.grid(
      interaction.depth = 3,
      n.trees = 100,
      shrinkage = 0.1,
      n.minobsinnode = 10
    )

    model <- caret::train(
      x = X,
      y = y,
      method = "gbm",
      metric = "ROC",
      trControl = ctrl,
      verbose = FALSE,
      tuneGrid = grid,
      bag.fraction = 0.5
    )
  }

  return(model)
}


# ------------------------------------------------------------------------------
# Evaluation, Running, and Ensemble Functions
# ------------------------------------------------------------------------------
#' @title Evaluate Diagnostic Model Performance
#' @description Evaluates the performance of a trained diagnostic model using
#'   various metrics relevant to binary classification, including AUROC, AUPRC,
#'   and metrics at an optimal or specified probability threshold.
#'
#' @param model_obj A trained model object (typically a `caret::train` object
#'   or a list from an ensemble like Bagging). Can be `NULL` if `precomputed_prob` is provided.
#' @param X_data A data frame of features corresponding to the data used for evaluation.
#'   Required if `model_obj` is provided and `precomputed_prob` is `NULL`.
#' @param y_data A factor vector of true class labels for the evaluation data.
#' @param sample_ids A vector of sample IDs for the evaluation data.
#' @param threshold_choices A character string specifying the thresholding strategy
#'   ("default", "f1", "youden") or a numeric probability threshold value (0-1).
#' @param pos_class A character string, the label for the positive class.
#' @param neg_class A character string, the label for the negative class.
#' @param precomputed_prob Optional. A numeric vector of precomputed probabilities
#'   for the positive class. If provided, `model_obj` and `X_data` are not used
#'   for score derivation.
#' @param y_original_numeric Optional. The original numeric/character vector of labels.
#'   If not provided, it's inferred from `y_data` using global `pos_label_value` and `neg_label_value`.
#'
#' @return A list containing:
#'   \itemize{
#'     \item `sample_score`: A data frame with `sample` (ID), `label` (original numeric),
#'       and `score` (predicted probability for positive class).
#'     \item `evaluation_metrics`: A list of performance metrics:
#'       \itemize{
#'         \item `Threshold_Strategy`: The strategy used for threshold selection.
#'         \item `_Threshold`: The chosen probability threshold.
#'         \item `Accuracy`, `Precision`, `Recall`, `F1`, `Specificity`: Metrics
#'           calculated at `_Threshold`.
#'         \item `AUROC`: Area Under the Receiver Operating Characteristic curve.
#'         \item `AUROC_95CI_Lower`, `AUROC_95CI_Upper`: 95% confidence interval for AUROC.
#'         \item `AUPRC`: Area Under the Precision-Recall curve.
#'       }
#'   }
#' @examples
#' \donttest{
#' set.seed(42)
#' n_obs <- 50
#' X_toy <- data.frame(
#'   FeatureA = rnorm(n_obs),
#'   FeatureB = runif(n_obs, 0, 100)
#' )
#' y_toy <- factor(sample(c("Control", "Case"), n_obs, replace = TRUE),
#'                 levels = c("Control", "Case"))
#' ids_toy <- paste0("Sample", 1:n_obs)
#'
#' # 2. Train a model
#' rf_model <- rf_dia(X_toy, y_toy)
#'
#' # 3. Evaluate the model using F1-score optimal threshold
#' eval_results <- evaluate_model_dia(
#'   model_obj = rf_model,
#'   X_data = X_toy,
#'   y_data = y_toy,
#'   sample_ids = ids_toy,
#'   threshold_choices = "f1",
#'   pos_class = "Case",
#'   neg_class = "Control"
#' )
#' str(eval_results)
#' }
#' @importFrom caret predict.train
#' @importFrom pROC roc auc ci.auc
#' @importFrom PRROC pr.curve
#' @export
evaluate_model_dia <- function(model_obj = NULL, X_data = NULL, y_data, sample_ids,
                               threshold_choices = "default", pos_class, neg_class,
                               precomputed_prob = NULL, y_original_numeric = NULL) {

  y_data <- base::factor(y_data, levels = c(neg_class, pos_class))
  prob <- precomputed_prob

  if (is.null(prob)) {
    if (is.null(model_obj)) {
      stop("Either 'model_obj' or 'precomputed_prob' must be provided for evaluation.")
    }
    if (is.null(X_data)) {
      stop("X_data must be provided when deriving probabilities from 'model_obj'.")
    }

    if ("train" %in% class(model_obj)) {
      # Ensure new data has same columns as training data, in same order
      train_features <- names(model_obj$trainingData)[-ncol(model_obj$trainingData)]
      missing_features <- setdiff(train_features, names(X_data))
      if (length(missing_features) > 0) {
        stop(paste("Evaluation data is missing features that were present in training data:", paste(missing_features, collapse = ", ")))
      }
      X_data_ordered <- X_data[, train_features, drop = FALSE]
      prob <- caret::predict.train(model_obj, X_data_ordered, type = "prob")[, pos_class]
    } else if (is.list(model_obj) && !is.null(model_obj$model_type) && model_obj$model_type %in% c("bagging", "easyensemble")) {
      all_probs <- list()
      # For ensemble models, we need a consistent feature set for all base models
      first_base_model <- NULL
      for(m in model_obj$base_model_objects) {
        if(!is.null(m) && "train" %in% class(m)) {
          first_base_model <- m
          break
        }
      }
      if(is.null(first_base_model)) stop("No valid base models found in the ensemble to extract feature names.")
      train_features <- names(first_base_model$trainingData)[-ncol(first_base_model$trainingData)]

      missing_features <- setdiff(train_features, names(X_data))
      if (length(missing_features) > 0) {
        stop(paste("Evaluation data is missing features that were present in base models training data:", paste(missing_features, collapse = ", ")))
      }
      X_data_ordered <- X_data[, train_features, drop = FALSE]

      for (i in seq_along(model_obj$base_model_objects)) {
        current_model <- model_obj$base_model_objects[[i]]
        if (!is.null(current_model) && "train" %in% class(current_model)) {
          all_probs[[i]] <- caret::predict.train(current_model, X_data_ordered, type = "prob")[, pos_class]
        } else {
          warning(sprintf("Base model #%d is not a valid caret train object. Skipping.", i))
        }
      }
      if (length(all_probs) > 0 && any(!sapply(all_probs, is.null))) {
        prob_matrix <- do.call(cbind, all_probs)
        prob <- rowMeans(prob_matrix, na.rm = TRUE)
      } else {
        stop("No valid predictions from base models in ensemble.")
      }
    } else if (is.list(model_obj) && !is.null(model_obj$model_type) && model_obj$model_type == "stacking") {
      # This path should ideally not be hit as stacking eval should pass precomputed_prob
      # but including for completeness if needed elsewhere.
      stop("Stacking ensemble prediction within evaluate_model_dia is not directly supported without precomputed_prob. Please provide precomputed_prob.")
    } else {
      stop("Unsupported model type for prediction. Please provide a caret 'train' object, a 'bagging' result object, or 'precomputed_prob'.")
    }
  }

  if (!is.numeric(prob) || all(is.na(prob))) {
    stop("Probabilities for evaluation are invalid (not numeric or all NA).")
  }

  # Replace NA probabilities with median
  prob[is.na(prob)] <- stats::median(prob, na.rm = TRUE)

  if (is.null(y_original_numeric)) {
    y_original_numeric <- base::ifelse(y_data == pos_class, .model_registry_env_dia$pos_label_value, .model_registry_env_dia$neg_label_value)
  }

  sample_score_df <- data.frame(
    sample = sample_ids,
    label = y_original_numeric,
    score = prob
  )

  final_threshold <- 0.5 # Default value
  strategy_used <- "default" # Default strategy

  if (is.character(threshold_choices) && length(threshold_choices) == 1) {
    strategy <- tolower(threshold_choices)
    if (strategy == "f1") {
      final_threshold <- find_optimal_threshold_dia(prob, y_data, type = "f1", pos_class = pos_class, neg_class = neg_class)
      strategy_used <- "f1"
    } else if (strategy == "youden") {
      final_threshold <- find_optimal_threshold_dia(prob, y_data, type = "youden", pos_class = pos_class, neg_class = neg_class)
      strategy_used <- "youden"
    } else if (strategy == "default") {
      final_threshold <- 0.5
      strategy_used <- "default"
    } else {
      warning(sprintf("Invalid string for threshold_choices: '%s'. Falling back to default (0.5).", threshold_choices))
    }
  } else if (is.numeric(threshold_choices) && length(threshold_choices) == 1 && threshold_choices >= 0 && threshold_choices <= 1) {
    final_threshold <- threshold_choices
    strategy_used <- "numeric"
  } else {
    warning("Invalid threshold_choices value. It must be 'f1', 'youden', 'default', or a numeric value between 0 and 1. Falling back to default (0.5).")
  }

  metrics_at_threshold <- calculate_metrics_at_threshold_dia(prob, y_data, final_threshold, pos_class = pos_class, neg_class = neg_class)

  roc_obj <- pROC::roc(y_data, prob, quiet = TRUE, levels = c(neg_class, pos_class))
  roc_auc <- pROC::auc(roc_obj)
  roc_ci_lower <- NA ; roc_ci_upper <- NA
  tryCatch({
    roc_ci <- pROC::ci.auc(roc_obj, conf.level = 0.95)
    roc_ci_lower <- roc_ci[1]
    roc_ci_upper <- roc_ci[3]
  }, error = function(e) {
    warning(paste("Could not calculate ROC CI for this model:", e$message))
  })

  pr_curve_obj <- PRROC::pr.curve(scores.class0 = prob, weights.class0 = as.numeric(y_data == pos_class))
  pr_auc <- pr_curve_obj$auc.integral

  evaluation_metrics <- list(
    Threshold_Strategy = strategy_used,
    Final_Threshold = final_threshold,
    Accuracy = metrics_at_threshold$Accuracy,
    Precision = metrics_at_threshold$Precision,
    Recall = metrics_at_threshold$Recall,
    F1 = metrics_at_threshold$F1,
    Specificity = metrics_at_threshold$Specificity,
    AUROC = roc_auc,
    AUROC_95CI_Lower = roc_ci_lower,
    AUROC_95CI_Upper = roc_ci_upper,
    AUPRC = pr_auc
  )

  return(list(sample_score = sample_score_df, evaluation_metrics = evaluation_metrics))
}

#' @title Run Multiple Diagnostic Models
#' @description Trains and evaluates one or more registered diagnostic models on a given dataset.
#'
#' @param data A data frame where the first column is the sample ID, the second
#'   is the outcome label, and subsequent columns are features.
#' @param model A character string or vector of character strings, specifying
#'   which models to run. Use "all_dia" to run all registered models.
#' @param tune Logical, whether to enable hyperparameter tuning for individual models.
#' @param seed An integer, for reproducibility of random processes.
#' @param threshold_choices A character string (e.g., "f1", "youden", "default")
#'   or a numeric value (0-1), or a named list/vector allowing different threshold
#'   strategies/values for each model.
#' @param positive_label_value A numeric or character value in the raw data
#'   representing the positive class.
#' @param negative_label_value A numeric or character value in the raw data
#'   representing the negative class.
#' @param new_positive_label A character string, the desired factor level name
#'   for the positive class (e.g., "Positive").
#' @param new_negative_label A character string, the desired factor level name
#'   for the negative class (e.g., "Negative").
#'
#' @return A named list, where each element corresponds to a run model and
#'   contains its trained `model_object`, `sample_score` data frame, and
#'   `evaluation_metrics`.
#' @examples
#' \donttest{
#' # This example assumes your package includes a dataset named 'train_dia'.
#' # If not, you should create a toy data frame similar to the one below.
#' #
#' # train_dia <- data.frame(
#' #   ID = paste0("Patient", 1:100),
#' #   Disease_Status = sample(c(0, 1), 100, replace = TRUE),
#' #   FeatureA = rnorm(100),
#' #   FeatureB = runif(100)
#' # )
#'
#' # Ensure the 'train_dia' dataset is available in the environment
#' # For example, if it is exported by your package:
#' # data(train_dia)
#'
#' # Check if 'train_dia' exists, otherwise skip the example
#' if (exists("train_dia")) {
#'   # 1. Initialize the modeling system
#'   initialize_modeling_system_dia()
#'
#'   # 2. Run selected models
#'   results <- models_dia(
#'     data = train_dia,
#'     model = c("rf", "lasso"), # Run only Random Forest and Lasso
#'     threshold_choices = list(rf = "f1", lasso = 0.6), # Different thresholds
#'     positive_label_value = 1,
#'     negative_label_value = 0,
#'     new_positive_label = "Case",
#'     new_negative_label = "Control",
#'     seed = 42
#'   )
#'
#'   # 3. Print summaries
#'   for (model_name in names(results)) {
#'     print_model_summary_dia(model_name, results[[model_name]])
#'   }
#' }
#' }
#' @seealso \code{\link{initialize_modeling_system_dia}}, \code{\link{evaluate_model_dia}}
#' @export
models_dia <- function(data,
                       model = "all_dia",
                       tune = FALSE,
                       seed = 123,
                       threshold_choices = "default",
                       positive_label_value = 1,
                       negative_label_value = 0,
                       new_positive_label = "Positive",
                       new_negative_label = "Negative") {

  if (!.model_registry_env_dia$is_initialized) {
    stop("Modeling system not initialized. Please call 'initialize_modeling_system_dia()' first.")
  }

  .model_registry_env_dia$pos_label_value <- positive_label_value
  .model_registry_env_dia$neg_label_value <- negative_label_value

  all_registered_models <- get_registered_models_dia()

  models_to_run_names <- NULL
  if (length(model) == 1 && model == "all_dia") {
    models_to_run_names <- names(all_registered_models)
  } else if (all(model %in% names(all_registered_models))) {
    models_to_run_names <- model
  } else {
    stop(paste("Invalid model name(s) provided. Available models are:", paste(names(all_registered_models), collapse = ", ")))
  }

  set.seed(seed)
  data_prepared <- .prepare_data_dia(data,
                                     positive_label_value, negative_label_value,
                                     new_positive_label, new_negative_label)

  X_data <- data_prepared$X
  y_data <- data_prepared$y
  sample_ids <- data_prepared$sample_ids
  pos_label_used <- data_prepared$pos_class_label
  neg_label_used <- data_prepared$neg_class_label
  y_original_numeric <- data_prepared$y_original_numeric

  # Determine threshold settings for each model
  model_thresholds <- list()
  if (is.list(threshold_choices) || (is.vector(threshold_choices) && !is.null(names(threshold_choices)))) {
    # A named list/vector is provided for model-specific thresholds
    for (m_name in models_to_run_names) {
      if (m_name %in% names(threshold_choices)) {
        model_thresholds[[m_name]] <- threshold_choices[[m_name]]
      } else {
        model_thresholds[[m_name]] <- "default" # Fallback for models not in the list
      }
    }
  } else {
    # A single global threshold choice is provided for all models
    for (m_name in models_to_run_names) {
      model_thresholds[[m_name]] <- threshold_choices
    }
  }

  all_model_results <- list()

  for (model_name in models_to_run_names) {
    current_model_func <- get_registered_models_dia()[[model_name]]
    current_threshold <- model_thresholds[[model_name]]

    message(sprintf("Running model: %s", model_name))

    mdl <- tryCatch({
      set.seed(seed)
      current_model_func(X_data, y_data, tune = tune)
    }, error = function(e) {
      warning(paste("Model", model_name, "failed during training:", conditionMessage(e)))
      NULL
    })

    if (!is.null(mdl)) {
      eval_results <- tryCatch({
        evaluate_model_dia(model_obj = mdl, X_data = X_data, y_data = y_data, sample_ids = sample_ids,
                           threshold_choices = current_threshold,
                           pos_class = pos_label_used,
                           neg_class = neg_label_used,
                           y_original_numeric = y_original_numeric)
      }, error = function(e) {
        warning(paste("Model", model_name, "failed during evaluation:", conditionMessage(e)))
        list(sample_score = data.frame(sample = sample_ids, label = y_original_numeric, score = NA),
             evaluation_metrics = list(error = paste("Evaluation failed:", conditionMessage(e))))
      })

      all_model_results[[model_name]] <- list(
        model_object = mdl,
        sample_score = eval_results$sample_score,
        evaluation_metrics = eval_results$evaluation_metrics
      )
    } else {
      failed_sample_score <- data.frame(
        sample = sample_ids,
        label = y_original_numeric,
        score = NA
      )
      all_model_results[[model_name]] <- list(
        model_object = NULL,
        sample_score = failed_sample_score,
        evaluation_metrics = list(error = "Model training failed.")
      )
    }
  }

  return(all_model_results)
}


#' @title Train a Bagging Diagnostic Model
#' @description Implements a Bagging (Bootstrap Aggregating) ensemble for
#'   diagnostic models. It trains multiple base models on bootstrapped samples
#'   of the training data and aggregates their predictions by averaging probabilities.
#'
#' @param data A data frame where the first column is the sample ID, the second
#'   is the outcome label, and subsequent columns are features.
#' @param base_model_name A character string, the name of the base diagnostic
#'   model to use (e.g., "rf", "lasso"). This model must be registered.
#' @param n_estimators An integer, the number of base models to train.
#' @param subset_fraction A numeric value between 0 and 1, the fraction of
#'   samples to bootstrap for each base model.
#' @param tune_base_model Logical, whether to enable tuning for each base model.
#' @param threshold_choices A character string (e.g., "f1", "youden", "default")
#'   or a numeric value (0-1) for determining the evaluation threshold for the ensemble.
#' @param positive_label_value A numeric or character value in the raw data
#'   representing the positive class.
#' @param negative_label_value A numeric or character value in the raw data
#'   representing the negative class.
#' @param new_positive_label A character string, the desired factor level name
#'   for the positive class (e.g., "Positive").
#' @param new_negative_label A character string, the desired factor level name
#'   for the negative class (e.g., "Negative").
#' @param seed An integer, for reproducibility.
#'
#' @return A list containing the `model_object`, `sample_score`, and `evaluation_metrics`.
#' @examples
#' \donttest{
#' # This example assumes your package includes a dataset named 'train_dia'.
#' # If not, create a toy data frame first.
#' if (exists("train_dia")) {
#'   initialize_modeling_system_dia()
#'
#'   bagging_rf_results <- bagging_dia(
#'     data = train_dia,
#'     base_model_name = "rf",
#'     n_estimators = 5, # Reduced for a quick example
#'     threshold_choices = "youden",
#'     positive_label_value = 1,
#'     negative_label_value = 0,
#'     new_positive_label = "Case",
#'     new_negative_label = "Control"
#'   )
#'   print_model_summary_dia("Bagging (RF)", bagging_rf_results)
#' }
#' }
#' @seealso \code{\link{initialize_modeling_system_dia}}, \code{\link{evaluate_model_dia}}
#' @export
bagging_dia <- function(data,
                        base_model_name,
                        n_estimators = 50,
                        subset_fraction = 0.632,
                        tune_base_model = FALSE,
                        threshold_choices = "default",
                        positive_label_value = 1,
                        negative_label_value = 0,
                        new_positive_label = "Positive",
                        new_negative_label = "Negative",
                        seed = 456) {

  if (!.model_registry_env_dia$is_initialized) {
    initialize_modeling_system_dia()
  }
  .model_registry_env_dia$pos_label_value <- positive_label_value
  .model_registry_env_dia$neg_label_value <- negative_label_value

  all_registered_models <- get_registered_models_dia()
  if (!(base_model_name %in% names(all_registered_models))) {
    stop(sprintf("Base model '%s' not found. Please register it first.", base_model_name))
  }

  message(sprintf("Running Bagging model: %s (base: %s)", "Bagging_dia", base_model_name))

  set.seed(seed)
  data_prepared <- .prepare_data_dia(data,
                                     positive_label_value, negative_label_value,
                                     new_positive_label, new_negative_label)

  X_data <- data_prepared$X
  y_data <- data_prepared$y
  sample_ids <- data_prepared$sample_ids
  pos_label_used <- data_prepared$pos_class_label
  neg_label_used <- data_prepared$neg_class_label
  y_original_numeric <- data_prepared$y_original_numeric

  n_samples <- nrow(X_data)
  subset_size <- base::floor(n_samples * subset_fraction)
  if (subset_size == 0) stop("Subset size is 0. Please check your data or subset_fraction.")

  trained_models_and_probs <- list()
  base_model_func <- get_registered_models_dia()[[base_model_name]]
  base_model_train_features <- NULL

  for (i in 1:n_estimators) {
    set.seed(seed + i)
    indices <- sample(1:n_samples, subset_size, replace = TRUE)
    X_boot <- X_data[indices, , drop = FALSE]
    y_boot <- y_data[indices]

    if (length(unique(y_boot)) < 2) {
      warning(sprintf("Bootstrap sample %d has only one class. Skipping this model.", i))
      trained_models_and_probs[[i]] <- list(model = NULL, prob = rep(NA, n_samples))
      next
    }

    current_model <- tryCatch({
      base_model_func(X_boot, y_boot, tune = tune_base_model)
    }, error = function(e) {
      warning(sprintf("Training base model %s for bootstrap %d failed: %s", base_model_name, i, e$message))
      NULL
    })

    prob_on_full_data <- rep(NA, n_samples)
    if (!is.null(current_model)) {
      tryCatch({
        prob_on_full_data <- caret::predict.train(current_model, X_data, type = "prob")[, pos_label_used]
      }, error = function(e) {
        warning(sprintf("Prediction for base model %s for bootstrap %d failed: %s", base_model_name, i, e$message))
      })
    }
    trained_models_and_probs[[i]] <- list(model = current_model, prob = prob_on_full_data)
  }

  valid_models <- lapply(trained_models_and_probs, `[[`, "model")
  valid_probs_list <- lapply(trained_models_and_probs, `[[`, "prob")
  valid_models <- valid_models[!sapply(valid_models, is.null)]
  valid_probs_list <- valid_probs_list[!sapply(valid_probs_list, function(p) all(is.na(p)))]

  if (length(valid_probs_list) == 0) {
    stop("No base models were successfully trained or made valid predictions. Cannot perform bagging.")
  }

  aggregated_prob <- rowMeans(do.call(cbind, valid_probs_list), na.rm = TRUE)

  bagging_model_obj_for_eval <- list(
    model_type = "bagging",
    base_model_name = base_model_name,
    n_estimators = n_estimators,
    base_model_objects = valid_models
  )

  eval_results <- evaluate_model_dia(
    model_obj = bagging_model_obj_for_eval,
    X_data = X_data,
    y_data = y_data, sample_ids = sample_ids,
    threshold_choices = threshold_choices,
    pos_class = pos_label_used, neg_class = neg_label_used,
    precomputed_prob = aggregated_prob,
    y_original_numeric = y_original_numeric
  )

  list(
    model_object = bagging_model_obj_for_eval,
    sample_score = eval_results$sample_score,
    evaluation_metrics = eval_results$evaluation_metrics
  )
}


#' @title Train a Stacking Diagnostic Model
#' @description Implements a Stacking ensemble. It trains multiple base models,
#'   then uses their predictions as features to train a meta-model.
#'
#' @param results_all_models A list of results from `models_dia()`,
#'   containing trained base model objects and their evaluation metrics.
#' @param data A data frame where the first column is the sample ID, the second
#'   is the outcome label, and subsequent columns are features. Used for training the meta-model.
#' @param meta_model_name A character string, the name of the meta-model to use
#'   (e.g., "lasso", "gbm"). This model must be registered.
#' @param top An integer, the number of top-performing base models (ranked by AUROC)
#'   to select for the stacking ensemble.
#' @param tune_meta Logical, whether to enable tuning for the meta-model.
#' @param threshold_choices A character string (e.g., "f1", "youden", "default")
#'   or a numeric value (0-1) for determining the evaluation threshold for the ensemble.
#' @param seed An integer, for reproducibility.
#' @param positive_label_value A numeric or character value in the raw data
#'   representing the positive class.
#' @param negative_label_value A numeric or character value in the raw data
#'   representing the negative class.
#' @param new_positive_label A character string, the desired factor level name
#'   for the positive class (e.g., "Positive").
#' @param new_negative_label A character string, the desired factor level name
#'   for the negative class (e.g., "Negative").
#'
#' @return A list containing the `model_object`, `sample_score`, and `evaluation_metrics`.
#' @examples
#' \donttest{
#' # 1. Initialize the modeling system
#' initialize_modeling_system_dia()
#'
#' # 2. Create a toy dataset for demonstration
#' set.seed(42)
#' data_toy <- data.frame(
#'   ID = paste0("Sample", 1:60),
#'   Status = sample(c(0, 1), 60, replace = TRUE),
#'   Feat1 = rnorm(60),
#'   Feat2 = runif(60)
#' )
#'
#' # 3. Generate mock base model results (as if from models_dia)
#' # In a real scenario, you would run models_dia() on your full dataset
#' base_model_results <- models_dia(
#'   data = data_toy,
#'   model = c("rf", "lasso"),
#'   seed = 123
#' )
#'
#' # 4. Run the stacking ensemble
#' stacking_results <- stacking_dia(
#'   results_all_models = base_model_results,
#'   data = data_toy,
#'   meta_model_name = "gbm",
#'   top = 2,
#'   threshold_choices = "f1"
#' )
#' print_model_summary_dia("Stacking (GBM)", stacking_results)
#' }
#' @importFrom dplyr select left_join
#' @importFrom magrittr %>%
#' @seealso \code{\link{models_dia}}, \code{\link{evaluate_model_dia}}
#' @export
stacking_dia <- function(results_all_models, data,
                         meta_model_name, top = 5, tune_meta = FALSE, threshold_choices = "f1", seed = 789,
                         positive_label_value = 1, negative_label_value = 0,
                         new_positive_label = "Positive", new_negative_label = "Negative") {

  if (!.model_registry_env_dia$is_initialized) {
    stop("Modeling system not initialized. Please call 'initialize_modeling_system_dia()' first.")
  }
  .model_registry_env_dia$pos_label_value <- positive_label_value
  .model_registry_env_dia$neg_label_value <- negative_label_value

  all_registered_models <- get_registered_models_dia()
  if (!(meta_model_name %in% names(all_registered_models))) {
    stop(sprintf("Meta-model '%s' not found. Please register it first.", meta_model_name))
  }

  message(sprintf("Running Stacking model: %s (meta: %s)", "Stacking_dia", meta_model_name))

  set.seed(seed)
  data_prepared <- .prepare_data_dia(data,
                                     positive_label_value, negative_label_value,
                                     new_positive_label, new_negative_label)

  y_true <- data_prepared$y
  sample_ids <- data_prepared$sample_ids
  pos_class <- data_prepared$pos_class_label
  neg_class <- data_prepared$neg_class_label
  y_original_numeric <- data_prepared$y_original_numeric

  model_aurocs <- sapply(results_all_models, function(res) res$evaluation_metrics$AUROC %||% NA)
  model_aurocs <- model_aurocs[!is.na(model_aurocs)]
  if (length(model_aurocs) == 0) stop("No base models with valid AUROC found for stacking.")

  sorted_models_names <- names(sort(model_aurocs, decreasing = TRUE))
  selected_base_models_names <- utils::head(sorted_models_names, min(top, length(sorted_models_names)))
  if (length(selected_base_models_names) < 1) stop("No base models selected for stacking.")

  selected_base_model_objects <- lapply(results_all_models[selected_base_models_names], `[[`, "model_object")

  # Prepare meta-features
  all_scores <- lapply(results_all_models[selected_base_models_names], function(res) {
    res$sample_score[, c("sample", "score")]
  })

  X_meta <- Reduce(function(df1, df2) dplyr::left_join(df1, df2, by = "sample"), all_scores)
  names(X_meta) <- c("sample", paste0("pred_", selected_base_models_names))
  X_meta_features <- dplyr::select(X_meta, -sample)

  # Ensure all meta-features are numeric
  X_meta_features[] <- lapply(X_meta_features, as.numeric)

  meta_model_func <- all_registered_models[[meta_model_name]]
  meta_mdl <- tryCatch({
    set.seed(seed)
    meta_model_func(X_meta_features, y_true, tune = tune_meta)
  }, error = function(e) {
    stop(paste("Meta-model", meta_model_name, "failed with error:", conditionMessage(e)))
  })

  eval_results <- evaluate_model_dia(model_obj = meta_mdl, X_data = X_meta_features, y_data = y_true, sample_ids = sample_ids,
                                     threshold_choices = threshold_choices,
                                     pos_class = pos_class, neg_class = neg_class,
                                     y_original_numeric = y_original_numeric)

  stacking_model_obj <- list(
    model_type = "stacking", meta_model_name = meta_model_name,
    base_models_used = selected_base_models_names,
    base_model_objects = selected_base_model_objects,
    trained_meta_model = meta_mdl
  )

  list(
    model_object = stacking_model_obj,
    sample_score = eval_results$sample_score,
    evaluation_metrics = eval_results$evaluation_metrics
  )
}


#' @title Train a Voting Ensemble Diagnostic Model
#' @description Implements a Voting ensemble, combining predictions from multiple
#'   base models through soft or hard voting.
#'
#' @param results_all_models A list of results from `models_dia()`,
#'   containing trained base model objects and their evaluation metrics.
#' @param data A data frame where the first column is the sample ID, the second
#'   is the outcome label, and subsequent columns are features. Used for evaluation.
#' @param type A character string, "soft" for weighted average of probabilities
#'   or "hard" for majority class voting.
#' @param weight_metric A character string, the metric to use for weighting
#'   base models in soft voting (e.g., "AUROC", "F1"). Ignored for hard voting.
#' @param top An integer, the number of top-performing base models (ranked by
#'   `weight_metric`) to include in the ensemble.
#' @param seed An integer, for reproducibility.
#' @param threshold_choices A character string (e.g., "f1", "youden", "default")
#'   or a numeric value (0-1) for determining the evaluation threshold for the ensemble.
#' @param positive_label_value A numeric or character value in the raw data
#'   representing the positive class.
#' @param negative_label_value A numeric or character value in the raw data
#'   representing the negative class.
#' @param new_positive_label A character string, the desired factor level name
#'   for the positive class (e.g., "Positive").
#' @param new_negative_label A character string, the desired factor level name
#'   for the negative class (e.g., "Negative").
#'
#' @return A list containing the `model_object`, `sample_score`, and `evaluation_metrics`.
#' @examples
#' \donttest{
#' # 1. Initialize the modeling system
#' initialize_modeling_system_dia()
#'
#' # 2. Create a toy dataset for demonstration
#' set.seed(42)
#' data_toy <- data.frame(
#'   ID = paste0("Sample", 1:60),
#'   Status = sample(c(0, 1), 60, replace = TRUE),
#'   Feat1 = rnorm(60),
#'   Feat2 = runif(60)
#' )
#'
#' # 3. Generate mock base model results (as if from models_dia)
#' base_model_results <- models_dia(
#'   data = data_toy,
#'   model = c("rf", "lasso"),
#'   seed = 123
#' )
#'
#' # 4. Run the soft voting ensemble
#' soft_voting_results <- voting_dia(
#'   results_all_models = base_model_results,
#'   data = data_toy,
#'   type = "soft",
#'   weight_metric = "AUROC",
#'   top = 2,
#'   threshold_choices = "f1"
#' )
#' print_model_summary_dia("Soft Voting", soft_voting_results)
#' }
#' @seealso \code{\link{models_dia}}, \code{\link{evaluate_model_dia}}
#' @export
voting_dia <- function(results_all_models, data,
                       type = c("soft", "hard"), weight_metric = "AUROC", top = 5, seed = 789,
                       threshold_choices = "f1",
                       positive_label_value = 1, negative_label_value = 0,
                       new_positive_label = "Positive", new_negative_label = "Negative") {

  type <- match.arg(type)
  set.seed(seed)
  .model_registry_env_dia$pos_label_value <- positive_label_value
  .model_registry_env_dia$neg_label_value <- negative_label_value

  message(sprintf("Running Voting model: %s (type: %s)", "Voting_dia", type))

  data_prepared <- .prepare_data_dia(data,
                                     positive_label_value, negative_label_value,
                                     new_positive_label, new_negative_label)

  y_true <- data_prepared$y
  sample_ids <- data_prepared$sample_ids
  pos_class <- data_prepared$pos_class_label
  neg_class <- data_prepared$neg_class_label
  n_samples <- length(y_true)
  y_original_numeric <- data_prepared$y_original_numeric

  model_metrics <- sapply(results_all_models, function(res) res$evaluation_metrics[[weight_metric]] %||% NA)
  model_metrics <- model_metrics[!is.na(model_metrics)]

  if (length(model_metrics) == 0) {
    stop(sprintf("No base models with valid '%s' metric found in 'results_all_models'.", weight_metric))
  }

  sorted_models_names <- names(sort(model_metrics, decreasing = TRUE))
  selected_base_models_names <- utils::head(sorted_models_names, min(top, length(sorted_models_names)))

  if (length(selected_base_models_names) < 1) stop("No base models selected for voting.")

  selected_base_model_objects <- list()
  base_model_thresholds <- list()
  for (model_name in selected_base_models_names) {
    selected_base_model_objects[[model_name]] <- results_all_models[[model_name]]$model_object
    base_model_thresholds[[model_name]] <- results_all_models[[model_name]]$evaluation_metrics$Final_Threshold
  }

  final_prob_predictions <- NULL
  if (type == "soft") {
    weights <- model_metrics[selected_base_models_names]
    weights[weights < 0] <- 0
    if (sum(weights, na.rm=TRUE) <= 0) {
      warning("Sum of weights is non-positive. Using equal weights."); weights[] <- 1
    }

    prob_matrix <- sapply(results_all_models[selected_base_models_names], function(res) res$sample_score$score)

    final_prob_predictions <- apply(prob_matrix, 1, function(sample_probs) {
      stats::weighted.mean(x = sample_probs, w = weights, na.rm = TRUE)
    })

  } else { # type == "hard"
    predictions <- sapply(selected_base_models_names, function(name) {
      threshold <- base_model_thresholds[[name]] %||% 0.5
      ifelse(results_all_models[[name]]$sample_score$score >= threshold, 1, 0)
    })
    # Majority vote. rowMeans > 0.5 means more 1s than 0s. Tie breaks to 1.
    final_prob_predictions <- ifelse(rowMeans(predictions, na.rm = TRUE) >= 0.5, 1, 0)
  }

  final_prob_predictions[is.na(final_prob_predictions)] <- 0.5

  eval_results <- evaluate_model_dia(y_data = y_true, sample_ids = sample_ids,
                                     threshold_choices = threshold_choices,
                                     pos_class = pos_class, neg_class = neg_class,
                                     precomputed_prob = final_prob_predictions,
                                     y_original_numeric = y_original_numeric)

  voting_model_obj <- list(
    model_type = "voting", voting_type = type,
    weight_metric = if (type == "soft") weight_metric else NULL,
    base_models_used = selected_base_models_names,
    base_model_objects = selected_base_model_objects,
    base_model_weights = if (type == "soft") weights else NULL,
    base_model_thresholds = if (type == "hard") base_model_thresholds else NULL
  )

  list(
    model_object = voting_model_obj,
    sample_score = eval_results$sample_score,
    evaluation_metrics = eval_results$evaluation_metrics
  )
}


#' @title Train an EasyEnsemble Model for Imbalanced Classification
#' @description Implements the EasyEnsemble algorithm. It trains multiple base
#'   models on balanced subsets of the data (by undersampling the majority class)
#'   and aggregates their predictions.
#'
#' @param data A data frame where the first column is the sample ID, the second
#'   is the outcome label, and subsequent columns are features.
#' @param base_model_name A character string, the name of the base diagnostic
#'   model to use (e.g., "xb", "rf"). This model must be registered.
#' @param n_estimators An integer, the number of base models to train (number of subsets).
#' @param tune_base_model Logical, whether to enable tuning for each base model.
#' @param threshold_choices A character string (e.g., "f1", "youden", "default")
#'   or a numeric value (0-1) for determining the evaluation threshold for the ensemble.
#' @param positive_label_value A numeric or character value in the raw data
#'   representing the positive class.
#' @param negative_label_value A numeric or character value in the raw data
#'   representing the negative class.
#' @param new_positive_label A character string, the desired factor level name
#'   for the positive class (e.g., "Positive").
#' @param new_negative_label A character string, the desired factor level name
#'   for the negative class (e.g., "Negative").
#' @param seed An integer, for reproducibility.
#'
#' @return A list containing the `model_object`, `sample_score`, and `evaluation_metrics`.
#' @examples
#' \donttest{
#' # 1. Initialize the modeling system
#' initialize_modeling_system_dia()
#'
#' # 2. Create an imbalanced toy dataset
#' set.seed(42)
#' n_obs <- 100
#' n_minority <- 10
#' data_imbalanced_toy <- data.frame(
#'   ID = paste0("Sample", 1:n_obs),
#'   Status = c(rep(1, n_minority), rep(0, n_obs - n_minority)),
#'   Feat1 = rnorm(n_obs),
#'   Feat2 = runif(n_obs)
#' )
#'
#' # 3. Run the EasyEnsemble algorithm
#' # n_estimators is reduced for a quick example
#' easyensemble_results <- imbalance_dia(
#'   data = data_imbalanced_toy,
#'   base_model_name = "rf",
#'   n_estimators = 3,
#'   threshold_choices = "f1"
#' )
#' print_model_summary_dia("EasyEnsemble (RF)", easyensemble_results)
#' }
#' @seealso \code{\link{initialize_modeling_system_dia}}, \code{\link{evaluate_model_dia}}
#' @export
imbalance_dia <- function(data,
                          base_model_name = "rf",
                          n_estimators = 10,
                          tune_base_model = FALSE,
                          threshold_choices = "default",
                          positive_label_value = 1,
                          negative_label_value = 0,
                          new_positive_label = "Positive",
                          new_negative_label = "Negative",
                          seed = 456) {

  if (!.model_registry_env_dia$is_initialized) {
    initialize_modeling_system_dia()
  }
  .model_registry_env_dia$pos_label_value <- positive_label_value
  .model_registry_env_dia$neg_label_value <- negative_label_value

  all_registered_models <- get_registered_models_dia()
  if (!(base_model_name %in% names(all_registered_models))) {
    stop(sprintf("Base model '%s' not found. Please register it first.", base_model_name))
  }

  message(sprintf("Running Imbalance model: %s (base: %s)", "EasyEnsemble_dia", base_model_name))

  set.seed(seed)
  data_prepared <- .prepare_data_dia(data,
                                     positive_label_value, negative_label_value,
                                     new_positive_label, new_negative_label)

  X_data <- data_prepared$X
  y_data <- data_prepared$y
  sample_ids <- data_prepared$sample_ids
  pos_label_used <- data_prepared$pos_class_label
  neg_label_used <- data_prepared$neg_class_label
  y_original_numeric <- data_prepared$y_original_numeric
  n_samples <- nrow(X_data)

  pos_indices <- which(y_data == pos_label_used)
  neg_indices <- which(y_data == neg_label_used)

  if (length(pos_indices) == 0 || length(neg_indices) == 0) {
    stop("Data has only one class. Cannot perform undersampling.")
  }

  is_pos_minority <- length(pos_indices) < length(neg_indices)
  minority_indices <- if (is_pos_minority) pos_indices else neg_indices
  majority_indices <- if (is_pos_minority) neg_indices else pos_indices
  min_size <- length(minority_indices)

  if (length(majority_indices) < min_size) {
    stop("Majority class has fewer samples than minority. Check data balance.")
  }

  base_model_func <- get_registered_models_dia()[[base_model_name]]
  all_probs <- matrix(NA, nrow = n_samples, ncol = n_estimators)
  valid_models <- list()

  for (i in 1:n_estimators) {
    set.seed(seed + i)
    sampled_majority <- sample(majority_indices, min_size, replace = FALSE)
    balanced_indices <- c(minority_indices, sampled_majority)
    X_bal <- X_data[balanced_indices, , drop = FALSE]
    y_bal <- y_data[balanced_indices]

    current_model <- tryCatch({
      base_model_func(X_bal, y_bal, tune = tune_base_model)
    }, error = function(e) {
      warning(sprintf("Training base model %s for subset %d failed: %s", base_model_name, i, e$message))
      NULL
    })

    if (!is.null(current_model)) {
      valid_models[[i]] <- current_model
      prob_on_full_data <- tryCatch({
        caret::predict.train(current_model, X_data, type = "prob")[, pos_label_used]
      }, error = function(e) {
        warning(sprintf("Prediction for base model %d failed: %s", i, e$message)); rep(NA, n_samples)
      })
      all_probs[, i] <- prob_on_full_data
    }
  }

  valid_models <- valid_models[!sapply(valid_models, is.null)]
  if (length(valid_models) == 0) stop("No base models were successfully trained.")

  aggregated_prob <- rowMeans(all_probs, na.rm = TRUE)

  easyensemble_model_obj <- list(
    model_type = "easyensemble", base_model_name = base_model_name,
    n_estimators = length(valid_models), base_model_objects = valid_models
  )

  eval_results <- evaluate_model_dia(
    model_obj = easyensemble_model_obj,
    X_data = X_data,
    y_data = y_data, sample_ids = sample_ids,
    threshold_choices = threshold_choices,
    pos_class = pos_label_used, neg_class = neg_label_used,
    precomputed_prob = aggregated_prob,
    y_original_numeric = y_original_numeric
  )

  list(
    model_object = easyensemble_model_obj,
    sample_score = eval_results$sample_score,
    evaluation_metrics = eval_results$evaluation_metrics
  )
}


#' @title Apply a Trained Model to New Data
#' @description Applies a trained diagnostic model (single or ensemble) to a new
#'   dataset to generate predictions. It can handle various model objects created
#'   by the package, including single caret models, Bagging, Stacking, Voting,
#'   and EasyEnsemble objects.
#'
#' @param trained_model_object A trained model object from `models_dia`,
#'   `bagging_dia`, `stacking_dia`, `voting_dia`, or `imbalance_dia`.
#' @param new_data A data frame containing the new samples for prediction.
#'   The first column must be the sample ID.
#' @param label_col_name An optional character string specifying the name of the
#'   column in `new_data` that contains the true labels. **If `NULL` (the default),
#'   the function will assume the second column is the label column.** To
#'   explicitly prevent label extraction (e.g., for data without labels), provide `NA`.
#' @param pos_class A character string for the positive class label used in the
#'   model's probability predictions. **Defaults to `"Positive"`.**
#' @param neg_class A character string for the negative class label. This parameter
#'   is mainly for consistency, as prediction focuses on `pos_class` probability.
#'   **Defaults to `"Negative"`.**
#'
#' @return A data frame with three columns: `sample` (the sample IDs), `label`
#'   (the true labels from `new_data`, or `NA` if not available/specified), and `score`
#'   (the predicted probability for the positive class).
#' @examples
#' \donttest{
#' # Assuming `bagging_results` and `test_dia` are available from previous steps
#' # bagging_model <- bagging_results$model_object
#'
#' # Example 1: Default behavior - use the second column of test_dia as label
#' # predictions <- apply_dia(
#' #   trained_model_object = bagging_model,
#' #   new_data = test_dia
#' # )
#'
#' # Example 2: Explicitly specify the label column by name
#' # predictions_explicit <- apply_dia(
#' #   trained_model_object = bagging_model,
#' #   new_data = test_dia,
#' #   label_col_name = "outcome"
#' # )
#'
#' # Example 3: Predict on data without labels
#' # test_data_no_labels <- test_dia[, -2] # Remove outcome column
#' # predictions_no_label <- apply_dia(
#' #   trained_model_object = bagging_model,
#' #   new_data = test_data_no_labels,
#' #   label_col_name = NA # Explicitly disable label extraction
#' # )
#' }
#' @importFrom stats median weighted.mean
#' @importFrom utils head
#' @export
apply_dia <- function(trained_model_object, new_data,
                      label_col_name = NULL,
                      pos_class = "Positive",
                      neg_class = "Negative") {
  # --- Input Validation ---
  if (!is.data.frame(new_data) || ncol(new_data) < 1) {
    stop("'new_data' must be a data frame with at least one column for sample IDs.")
  }
  if (is.null(trained_model_object)) {
    stop("'trained_model_object' cannot be NULL.")
  }

  sample_ids <- new_data[[1]]
  true_labels <- NA
  feature_col_names <- names(new_data)[-1] # Start with all columns except ID

  # --- Determine Label Column ---
  if (is.null(label_col_name)) {
    if (ncol(new_data) >= 2) {
      label_col_name <- names(new_data)[2]
      message(sprintf("`label_col_name` not provided, defaulting to the second column: '%s'. To disable, set label_col_name = NA.", label_col_name))
    } else {
      label_col_name <- NA
    }
  }

  if (!is.na(label_col_name)) {
    if (label_col_name %in% names(new_data)) {
      true_labels <- new_data[[label_col_name]]
      feature_col_names <- setdiff(feature_col_names, label_col_name)
    } else {
      warning(sprintf("Specified label column '%s' not found in 'new_data'. No labels will be extracted.", label_col_name))
    }
  }

  X_new <- new_data[, feature_col_names, drop = FALSE]

  # --- Predict Probabilities based on Model Type ---
  prob <- NULL
  model_type <- if ("train" %in% class(trained_model_object)) "caret" else trained_model_object$model_type

  if (is.null(model_type)) {
    stop("Unsupported model type. Please provide a caret 'train' object or a supported ensemble object.")
  }

  # Helper to get consistently ordered feature set
  get_ordered_features <- function(model_obj, new_data) {
    train_features <- names(model_obj$trainingData)[-ncol(model_obj$trainingData)]
    missing_features <- setdiff(train_features, names(new_data))
    if (length(missing_features) > 0) {
      stop(paste("New data is missing required features:", paste(missing_features, collapse = ", ")))
    }
    return(new_data[, train_features, drop = FALSE])
  }

  if (model_type == "caret") {
    X_new_ordered <- get_ordered_features(trained_model_object, X_new)
    prob <- caret::predict.train(trained_model_object, X_new_ordered, type = "prob")[, pos_class]
  } else if (model_type %in% c("bagging", "easyensemble")) {
    first_base_model <- trained_model_object$base_model_objects[[1]]
    if(is.null(first_base_model)) stop("No valid base models found in the ensemble.")
    X_new_ordered <- get_ordered_features(first_base_model, X_new)

    all_probs <- lapply(trained_model_object$base_model_objects, function(m) {
      if (!is.null(m)) caret::predict.train(m, X_new_ordered, type = "prob")[, pos_class] else NA
    })
    prob <- rowMeans(do.call(cbind, all_probs), na.rm = TRUE)
  } else if (model_type == "stacking") {
    X_meta_list <- lapply(trained_model_object$base_models_used, function(name) {
      base_model_obj <- trained_model_object$base_model_objects[[name]]
      X_new_ordered_base <- get_ordered_features(base_model_obj, X_new)
      caret::predict.train(base_model_obj, X_new_ordered_base, type = "prob")[, pos_class]
    })
    X_meta_features <- as.data.frame(do.call(cbind, X_meta_list))
    names(X_meta_features) <- paste0("pred_", trained_model_object$base_models_used)
    prob <- caret::predict.train(trained_model_object$trained_meta_model, X_meta_features, type = "prob")[, pos_class]
  } else if (model_type == "voting") {
    prob_matrix <- do.call(cbind, lapply(trained_model_object$base_models_used, function(name) {
      model_obj <- trained_model_object$base_model_objects[[name]]
      X_new_ordered_base <- get_ordered_features(model_obj, X_new)
      caret::predict.train(model_obj, X_new_ordered_base, type = "prob")[, pos_class]
    }))

    if (trained_model_object$voting_type == "soft") {
      prob <- apply(prob_matrix, 1, function(p) stats::weighted.mean(p, trained_model_object$base_model_weights, na.rm = TRUE))
    } else { # Hard voting
      class_matrix <- sapply(seq_along(trained_model_object$base_models_used), function(i) {
        threshold <- trained_model_object$base_model_thresholds[[i]] %||% 0.5
        ifelse(prob_matrix[, i] >= threshold, 1, 0)
      })
      prob <- rowMeans(class_matrix, na.rm = TRUE) # Returns proportion of positive votes
    }
  }

  if (is.null(prob)) stop("Failed to generate predictions.")
  prob[is.na(prob)] <- stats::median(prob, na.rm = TRUE)

  return(data.frame(sample = sample_ids, label = true_labels, score = prob))
}


#' @title Evaluate Predictions from a Data Frame
#' @description Evaluates model performance from a data frame of predictions,
#'   calculating metrics like AUROC, AUPRC, F1 score, etc. This function is designed
#'   for use with prediction results, such as the output from `apply_dia`.
#'
#' @param prediction_df  A data frame containing predictions. Must contain
#'   the columns `sample`, `label` (true labels), and `score` (predicted probabilities).
#' @param threshold_choices A character string specifying the thresholding strategy
#'   ("default", "f1", "youden") or a numeric probability threshold value (0-1).
#' @param pos_class A character string for the positive class label used in reporting.
#'   **Defaults to `"Positive"`.**
#' @param neg_class A character string for the negative class label used in reporting.
#'   **Defaults to `"Negative"`.**
#'
#' @details
#' This function strictly requires the `label` column in `prediction_df` to adhere
#' to the following format:
#' \itemize{
#'   \item **`1`**: Represents the positive class.
#'   \item **`0`**: Represents the negative class.
#'   \item **`NA`**: Will be ignored during calculation.
#' }
#' The function will stop with an error if any other values are found in the `label` column.
#'
#' @return A named list containing all calculated performance metrics.
#' @examples
#' \donttest{
#' # # Create a sample prediction data frame
#' # predictions_df <- data.frame(
#' #   sample = 1:10,
#' #   label = c(1, 0, 1, 1, 0, 0, 1, 0, 1, 0),
#' #   score = c(0.9, 0.2, 0.8, 0.6, 0.3, 0.4, 0.95, 0.1, 0.7, 0.5)
#' # )
#' #
#' # # Evaluate the predictions using the 'f1' threshold strategy
#' # evaluation_results <- evaluate_predictions_dia(
#' #   prediction_df = predictions_df,
#' #   threshold_choices = "f1"
#' # )
#' #
#' # print(evaluation_results)
#' }
#' @importFrom pROC roc auc ci.auc
#' @importFrom PRROC pr.curve
#' @export
evaluate_predictions_dia <- function(prediction_df,
                                     threshold_choices = "default",
                                     pos_class = "Positive",
                                     neg_class = "Negative") {
  # --- Input Validation ---
  required_cols <- c("sample", "label", "score")
  if (!all(required_cols %in% names(prediction_df))) {
    stop(sprintf("`prediction_df` must contain the columns: %s", paste(required_cols, collapse = ", ")))
  }
  if (all(is.na(prediction_df$label))) {
    stop("Cannot evaluate predictions: the 'label' column contains no true labels (all are NA).")
  }

  # --- Prepare and Validate Data for Evaluation ---
  prob <- prediction_df$score
  raw_labels <- prediction_df$label

  # Remove samples with missing labels or scores for calculation
  valid_indices <- !is.na(raw_labels) & !is.na(prob)
  if(sum(!valid_indices) > 0) {
    message(sprintf("%d rows with NA labels or scores were removed before evaluation.", sum(!valid_indices)))
  }
  raw_labels <- raw_labels[valid_indices]
  prob <- prob[valid_indices]

  if(length(raw_labels) == 0) stop("No valid pairs of labels and scores to evaluate.")

  # **NEW LOGIC: Strictly enforce that labels must be 0 or 1**
  unique_labels <- unique(raw_labels)
  if (!all(unique_labels %in% c(0, 1))) {
    invalid_labels <- unique_labels[!unique_labels %in% c(0, 1)]
    stop(sprintf("Invalid values found in the 'label' column: %s. This function requires labels to be strictly 0 (negative) or 1 (positive).",
                 paste(invalid_labels, collapse = ", ")))
  }
  if (length(unique_labels) < 2) {
    stop(paste("The 'label' column has fewer than two unique non-NA values (needs both 0 and 1):", paste(unique_labels, collapse=", ")))
  }

  # Create factor based on the 0/1 rule
  y_data <- factor(
    raw_labels,
    levels = c(0, 1),
    labels = c(neg_class, pos_class)
  )

  # --- Find Optimal Threshold ---
  final_threshold <- 0.5
  strategy_used <- "default"

  if (is.character(threshold_choices) && length(threshold_choices) == 1) {
    strategy <- tolower(threshold_choices)
    if (strategy == "f1") {
      final_threshold <- find_optimal_threshold_dia(prob, y_data, type = "f1", pos_class = pos_class, neg_class = neg_class)
      strategy_used <- "f1"
    } else if (strategy == "youden") {
      final_threshold <- find_optimal_threshold_dia(prob, y_data, type = "youden", pos_class = pos_class, neg_class = neg_class)
      strategy_used <- "youden"
    } else if (strategy != "default") {
      warning(sprintf("Invalid string for threshold_choices: '%s'. Falling back to default (0.5).", threshold_choices))
    }
  } else if (is.numeric(threshold_choices) && length(threshold_choices) == 1 && threshold_choices >= 0 && threshold_choices <= 1) {
    final_threshold <- threshold_choices
    strategy_used <- "numeric"
  } else if(!is.character(threshold_choices) || threshold_choices != "default"){
    warning("Invalid threshold_choices value. Falling back to default (0.5).")
  }

  # --- Calculate All Metrics ---
  metrics_at_threshold <- calculate_metrics_at_threshold_dia(prob, y_data, final_threshold, pos_class = pos_class, neg_class = neg_class)

  roc_obj <- pROC::roc(y_data, prob, quiet = TRUE, levels = c(neg_class, pos_class))
  roc_auc <- as.numeric(pROC::auc(roc_obj))
  roc_ci_lower <- NA ; roc_ci_upper <- NA
  tryCatch({
    roc_ci <- pROC::ci.auc(roc_obj, conf.level = 0.95)
    roc_ci_lower <- roc_ci[1]
    roc_ci_upper <- roc_ci[3]
  }, error = function(e) {
    warning(paste("Could not calculate ROC CI:", e$message))
  })

  pr_auc <- tryCatch({
    pr_curve_obj <- PRROC::pr.curve(scores.class0 = prob[y_data == pos_class], scores.class1 = prob[y_data == neg_class])
    pr_curve_obj$auc.integral
  }, error = function(e) {
    warning(paste("Could not calculate AUPRC:", e$message)); NA
  })

  list(
    Threshold_Strategy = strategy_used,
    Threshold = final_threshold,
    Accuracy = metrics_at_threshold$Accuracy,
    Precision = metrics_at_threshold$Precision,
    Recall = metrics_at_threshold$Recall,
    F1 = metrics_at_threshold$F1,
    Specificity = metrics_at_threshold$Specificity,
    AUROC = roc_auc,
    AUROC_95CI_Lower = roc_ci_lower,
    AUROC_95CI_Upper = roc_ci_upper,
    AUPRC = pr_auc
  )
}


#' @title Initialize Diagnostic Modeling System
#' @description Initializes the diagnostic modeling system by loading required
#'   packages and registering default diagnostic models (Random Forest, XGBoost,
#'   SVM, MLP, Lasso, Elastic Net, Ridge, LDA, QDA, Naive Bayes, Decision Tree, GBM).
#'   This function should be called once before using `models_dia()` or ensemble methods.
#'
#' @return Invisible NULL. Initializes the internal model registry.
#' @examples
#' \donttest{
#' # Initialize the system (typically run once at the start of a session or script)
#' initialize_modeling_system_dia()
#'
#' # Check if a default model like Random Forest is now registered
#' "rf" %in% names(get_registered_models_dia())
#' }
#' @export
initialize_modeling_system_dia <- function() {
  if (.model_registry_env_dia$is_initialized) {
    message("Diagnostic modeling system already initialized")
    return(invisible(NULL))
  }

  # Check core packages
  required_packages_dia <- c("caret", "pROC", "PRROC", "glmnet", "MASS", "gbm", "xgboost")

  missing <- required_packages_dia[!sapply(required_packages_dia, requireNamespace, quietly = TRUE)]

  if (length(missing) > 0) {
    stop(sprintf(
      "Missing required packages: %s\nPlease run: install.packages(c('%s'))",
      paste(missing, collapse = ", "),
      paste(missing, collapse = "', '")
    ))
  }

  # Register default models
  register_model_dia("rf", rf_dia)
  register_model_dia("xb", xb_dia)
  register_model_dia("svm", svm_dia)
  register_model_dia("mlp", mlp_dia)
  register_model_dia("lasso", lasso_dia)
  register_model_dia("en", en_dia)
  register_model_dia("ridge", ridge_dia)
  register_model_dia("lda", lda_dia)
  register_model_dia("qda", qda_dia)
  register_model_dia("nb", nb_dia)
  register_model_dia("dt", dt_dia)
  register_model_dia("gbm", gbm_dia)

  .model_registry_env_dia$is_initialized <- TRUE
  message("Diagnostic modeling system initialized and default models registered.")
  return(invisible(NULL))
}

#' @title Print Diagnostic Model Summary
#' @description Prints a formatted summary of the evaluation metrics for a
#'   diagnostic model, either from training data or new data evaluation.
#'
#' @param model_name A character string, the name of the model (e.g., "rf", "Bagging (RF)").
#' @param results_list A list containing model evaluation results, typically
#'   an element from the output of `models_dia()` or the result of `bagging_dia()`,
#'   `stacking_dia()`, `voting_dia()`, or `imbalance_dia()`. It must contain
#'   `evaluation_metrics` and `model_object` (if applicable).
#' @param on_new_data Logical, indicating whether the results are from applying
#'   the model to new, unseen data (`TRUE`) or from the training/internal validation
#'   data (`FALSE`).
#'
#' @return NULL. Prints the summary to the console.
#' @examples
#' # Example for a successfully evaluated model
#' successful_results <- list(
#'   evaluation_metrics = list(
#'     Threshold_Strategy = "f1",
#'     `_Threshold` = 0.45,
#'     AUROC = 0.85, AUROC_95CI_Lower = 0.75, AUROC_95CI_Upper = 0.95,
#'     AUPRC = 0.80, Accuracy = 0.82, F1 = 0.78,
#'     Precision = 0.79, Recall = 0.77, Specificity = 0.85
#'   )
#' )
#' print_model_summary_dia("MyAwesomeModel", successful_results)
#'
#' # Example for a failed model
#' failed_results <- list(evaluation_metrics = list(error = "Training failed"))
#' print_model_summary_dia("MyFailedModel", failed_results)
#' @export
print_model_summary_dia <- function(model_name, results_list, on_new_data = FALSE) {
  metrics <- results_list$evaluation_metrics
  model_info <- results_list$model_object

  if (!is.null(metrics$error)) {
    message(sprintf("Model: %-10s | Status: Failed (%s)", model_name, metrics$error))
  } else {
    data_source_str <- if(on_new_data) "on New Data" else "on Training Data"
    message(sprintf("\n--- %s Model (%s) Metrics ---", model_name, data_source_str))

    if (!is.null(model_info) && !is.null(model_info$model_type)) {
      if (model_info$model_type == "bagging") {
        message(sprintf("Ensemble Type: Bagging (Base: %s, Estimators: %d)",
                        model_info$base_model_name, model_info$n_estimators))
      } else if (model_info$model_type == "stacking") {
        message(sprintf("Ensemble Type: Stacking (Meta: %s, Base models used: %s)",
                        model_info$meta_model_name, paste(model_info$base_models_used, collapse = ", ")))
      } else if (model_info$model_type == "voting") {
        message(sprintf("Ensemble Type: Voting (Type: %s, Weight Metric: %s, Base models used: %s)",
                        model_info$voting_type,
                        if (!is.null(model_info$weight_metric)) model_info$weight_metric else "N/A",
                        paste(model_info$base_models_used, collapse = ", ")))
      } else if (model_info$model_type == "easyensemble") {
        message(sprintf("Ensemble Type: EasyEnsemble (Base: %s, Estimators: %d)",
                        model_info$base_model_name, model_info$n_estimators))
      }
    }

    message(sprintf("Threshold Strategy: %s (%.4f)", metrics$Threshold_Strategy, metrics$Final_Threshold))
    message(sprintf("AUROC: %.4f (95%% CI: %.4f - %.4f)",
                    metrics$AUROC, metrics$AUROC_95CI_Lower, metrics$AUROC_95CI_Upper))
    message(sprintf("AUPRC: %.4f", metrics$AUPRC))
    message(sprintf("Accuracy: %.4f", metrics$Accuracy))
    message(sprintf("F1: %.4f", metrics$F1))
    message(sprintf("Precision: %.4f", metrics$Precision))
    message(sprintf("Recall: %.4f", metrics$Recall))
    message(sprintf("Specificity: %.4f", metrics$Specificity))
    message("--------------------------------------------------")
  }
}
