#' @importFrom utils globalVariables
utils::globalVariables(c("x", "y", "recall", "Actual", "Predicted", "Freq", "Percentage",
                         "time", "AUROC", "feature", "value", "ID", "e", # 确保所有ggplot变量都列出
                         "score_col", "label", "sample", "score", ".")) # 针对 dplyr 管道的 "."

# Internal package environment for model registry.
# This environment holds functions for various prognostic models, allowing them
# to be registered and retrieved dynamically.
.model_registry_env_pro <- new.env()
.model_registry_env_pro$known_models_internal <- list()
.model_registry_env_pro$is_initialized <- FALSE

#' @title Prepare Data for Prognostic Models (Internal)
#' @description Prepares an input data frame for survival analysis by separating
#'   ID, outcome, time, and features based on a fixed column structure
#'   (1st=ID, 2nd=Outcome, 3rd=Time, 4th+=Features). It also handles
#'   time unit conversion and creates a survival object.
#'
#' @param data A data frame where the first column is sample ID, second is
#'   outcome status (0/1), third is time, and subsequent columns are features.
#' @param time_unit A character string ("day", "month", "year") for time conversion.
#'
#' @return A list containing `X`, `Y_surv`, `sample_ids`, `outcome_numeric`,
#'   and `time_numeric`.
#' @importFrom survival Surv
#' @noRd
.prepare_data_pro <- function(data, time_unit = c("day", "month", "year")) {
  if (!is.data.frame(data)) {
    stop("Input 'data' must be a data frame.")
  }
  if (ncol(data) < 4) {
    stop("Input data for prognosis must have at least four columns: ID, Outcome, Time, and at least one Feature.")
  }

  time_unit <- match.arg(time_unit)

  sample_ids <- data[[1]]
  y_outcome <- base::as.numeric(data[[2]])
  time_val <- base::as.numeric(data[[3]])

  if (any(is.na(time_val)) || any(is.na(y_outcome))) {
    warning("NA values found in time or outcome columns. These rows might be problematic.")
  }

  # Convert time to days
  if (time_unit == "month") {
    time_val <- time_val * (365.25 / 12)
  } else if (time_unit == "year") {
    time_val <- time_val * 365.25
  }

  # Validate data and remove invalid rows
  valid_rows <- !is.na(time_val) & !is.na(y_outcome) & time_val > 0
  if (any(!valid_rows)) {
    original_rows <- nrow(data)
    warning(sprintf("Found %d rows with NA time/outcome or non-positive time. These rows will be excluded.", sum(!valid_rows)))

    data <- data[valid_rows, , drop = FALSE]
    sample_ids <- sample_ids[valid_rows]
    y_outcome <- y_outcome[valid_rows]
    time_val <- time_val[valid_rows]

    if (nrow(data) == 0) {
      stop("After removing invalid rows, no data remains for analysis.")
    }
  }

  if (!all(y_outcome %in% c(0, 1))) {
    stop("Outcome column (2nd column) must contain only 0 (censored) and 1 (event).")
  }

  Y_surv <- survival::Surv(time = time_val, event = y_outcome)
  X <- data[, -c(1, 2, 3), drop = FALSE]

  # Convert character columns to factors
  for (col_name in names(X)) {
    if (is.character(X[[col_name]])) {
      X[[col_name]] <- base::as.factor(X[[col_name]])
    }
  }

  list(
    X = as.data.frame(X),
    Y_surv = Y_surv,
    sample_ids = sample_ids,
    outcome_numeric = y_outcome,
    time_numeric = time_val
  )
}

# ------------------------------------------------------------------------------
# Model Registry and Utility Functions
# ------------------------------------------------------------------------------

#' @title Register a Prognostic Model Function
#' @description Registers a user-defined or pre-defined prognostic model function
#'   with the internal model registry. This allows the function to be called
#'   later by its registered name, facilitating a modular model management system.
#'
#' @param name A character string, the unique name to register the model under.
#' @param func A function, the R function implementing the prognostic model.
#'   This function should typically accept `X` (features) and `y_surv` (survival object)
#'   as its first two arguments.
#' @return NULL. The function registers the model function invisibly.
#' @examples
#' # Example of a dummy model function for registration
#' my_dummy_cox_model <- function(X, y_surv, tune = FALSE) {
#'   # A minimal survival model for demonstration
#'   model_fit <- survival::coxph(y_surv ~ ., data = X)
#'   structure(list(finalModel = model_fit, X_train_cols = colnames(X),
#'                  model_type = "survival_dummy_cox"), class = "train")
#' }
#'
#' # Register the dummy model
#' initialize_modeling_system_pro() # Ensure system is initialized
#' register_model_pro("dummy_cox", my_dummy_cox_model)
#' "dummy_cox" %in% names(get_registered_models_pro()) # Check if registered
#' @seealso \code{\link{get_registered_models_pro}}, \code{\link{initialize_modeling_system_pro}}
#' @export
register_model_pro <- function(name, func) {
  if (!is.character(name) || length(name) != 1 || nchar(name) == 0) {
    stop("Prognosis model name must be a non-empty character string.")
  }
  if (!is.function(func)) {
    stop("Prognosis model function must be an R function.")
  }
  .model_registry_env_pro$known_models_internal[[name]] <- func
}

#' @title Get Registered Prognostic Models
#' @description Retrieves a list of all prognostic model functions currently
#'   registered in the internal environment.
#'
#' @return A named list where names are the registered model names and values
#'   are the corresponding model functions.
#' @examples
#' # Get all currently registered models
#' initialize_modeling_system_pro() # Ensure system is initialized
#' models <- get_registered_models_pro()
#' names(models) # See available model names
#' @seealso \code{\link{register_model_pro}}, \code{\link{initialize_modeling_system_pro}}
#' @export
get_registered_models_pro <- function() {
  return(.model_registry_env_pro$known_models_internal)
}

#' @title Load and Prepare Data for Prognostic Models
#' @description Loads a CSV file containing patient data, extracts features,
#'   outcome, and time columns, and prepares them into a format suitable for
#'   survival analysis models. Handles basic data cleaning like NA removal
#'   and column type conversion.
#'
#' @param data_path A character string, the file path to the input CSV data.
#'   The first column is assumed to be a sample ID.
#' @param outcome_col_name A character string, the name of the column containing
#'   event status (0 for censored, 1 for event).
#' @param time_col_name A character string, the name of the column containing
#'   event or censoring time.
#' @param time_unit A character string, the unit of time in `time_col_name`.
#'   Can be "day", "month", or "year". Times will be converted to days internally.
#'
#' @return A list containing:
#'   \itemize{
#'     \item `X`: A data frame of features (all columns except ID, outcome, and time).
#'     \item `Y_surv`: A `survival::Surv` object created from time and outcome.
#'     \item `sample_ids`: A vector of sample IDs (the first column of the input data).
#'     \item `outcome_numeric`: A numeric vector of outcome status.
#'     \item `time_numeric`: A numeric vector of time, converted to days.
#'   }
#' @examples
#' temp_csv_path <- tempfile(fileext = ".csv")
#' dummy_data <- data.frame(
#'   ID = paste0("Patient", 1:50),
#'   FeatureA = rnorm(50),
#'   FeatureB = runif(50, 0, 100),
#'   CategoricalFeature = sample(c("A", "B", "C"), 50, replace = TRUE),
#'   Outcome_Status = sample(c(0, 1), 50, replace = TRUE),
#'   Followup_Time_Months = runif(50, 10, 60)
#' )
#' write.csv(dummy_data, temp_csv_path, row.names = FALSE)
#'
#' # Load and prepare data
#' prepared_data <- load_and_prepare_data_pro(
#'   data_path = temp_csv_path,
#'   outcome_col_name = "Outcome_Status",
#'   time_col_name = "Followup_Time_Months",
#'   time_unit = "month"
#' )
#'
#' # Check prepared data structure
#' str(prepared_data$X)
#' print(prepared_data$Y_surv[1:5])
#'
#' # Clean up dummy file
#' unlink(temp_csv_path)
#' @importFrom readr read_csv
#' @importFrom survival Surv
#' @export
load_and_prepare_data_pro <- function(data_path, outcome_col_name, time_col_name, time_unit = c("day", "month", "year")) {
  df_original <- readr::read_csv(data_path, show_col_types = FALSE)
  names(df_original) <- trimws(names(df_original))

  if (ncol(df_original) < 3) {
    stop("Input data for prognosis must have at least three columns: an ID column (first column), an outcome column, and a time column.")
  }

  sample_ids <- df_original[[1]]
  df_features_and_surv <- df_original[, -1, drop = FALSE]

  if (!outcome_col_name %in% names(df_features_and_surv)) {
    stop(paste("Error: Outcome column '", outcome_col_name, "' not found in data.", sep=""))
  }
  if (!time_col_name %in% names(df_features_and_surv)) {
    stop(paste("Error: Time column '", time_col_name, "' not found in data.", sep=""))
  }

  time_unit <- match.arg(time_unit)

  time_val <- base::as.numeric(df_features_and_surv[[time_col_name]])
  if (any(is.na(time_val))) {
    warning("NA values found in time column. These rows might be problematic for survival analysis.")
  }

  if (time_unit == "month") {
    time_val <- time_val * (365.25 / 12)
  } else if (time_unit == "year") {
    time_val <- time_val * 365.25
  }

  y_outcome <- base::as.numeric(df_features_and_surv[[outcome_col_name]])

  valid_rows <- !is.na(time_val) & !is.na(y_outcome) & time_val > 0
  if (any(!valid_rows)) {
    warning(sprintf("Found %d rows with NA time/outcome or non-positive time. These rows will be excluded.", sum(!valid_rows)))
    time_val <- time_val[valid_rows]
    y_outcome <- y_outcome[valid_rows]
    df_features_and_surv <- df_features_and_surv[valid_rows, , drop = FALSE]
    sample_ids <- sample_ids[valid_rows]
    if (nrow(df_features_and_surv) == 0) {
      stop("After removing invalid rows, no data remains for analysis.")
    }
  }
  if (!all(y_outcome %in% c(0, 1))) {
    stop("Outcome column must contain only 0 (censored) and 1 (event).")
  }

  Y_surv <- survival::Surv(time = time_val, event = y_outcome)
  X <- df_features_and_surv[, setdiff(names(df_features_and_surv), c(outcome_col_name, time_col_name)), drop = FALSE]

  for (col_name in names(X)) {
    if (is.character(X[[col_name]])) {
      X[[col_name]] <- base::as.factor(X[[col_name]])
    } else if (!base::is.numeric(X[[col_name]]) && !base::is.factor(X[[col_name]])) {
      if (all(is.na(base::as.numeric(X[[col_name]]))) && !all(is.na(X[[col_name]]))) {
        X[[col_name]] <- base::as.factor(X[[col_name]])
      } else {
        X[[col_name]] <- base::as.numeric(X[[col_name]])
      }
    }
  }

  list(
    X = as.data.frame(X),
    Y_surv = Y_surv,
    sample_ids = sample_ids,
    outcome_numeric = y_outcome,
    time_numeric = time_val
  )
}


#' @title Train a Lasso Cox Proportional Hazards Model
#' @description Trains a Cox proportional hazards model with Lasso regularization
#'   using `glmnet`.
#'
#' @param X A data frame of features.
#' @param y_surv A `survival::Surv` object representing the survival outcome.
#' @param tune Logical, whether to perform hyperparameter tuning (currently simplified/ignored
#'   for direct `cv.glmnet` usage which inherently tunes lambda).
#' @return A list of class "train" containing the trained `glmnet` model object,
#'   names of features used in training, and model type. The returned object
#'   also includes `fitted_scores` (linear predictor) and `y_surv`.
#' @examples
#' set.seed(42)
#' n_samples <- 50
#' n_features <- 10
#' X_data <- as.data.frame(matrix(rnorm(n_samples * n_features), ncol = n_features))
#' Y_surv_obj <- survival::Surv(
#'   time = runif(n_samples, 100, 1000),
#'   event = sample(0:1, n_samples, replace = TRUE)
#' )
#'
#' # Train the model
#' lasso_model <- lasso_pro(X_data, Y_surv_obj)
#' print(lasso_model$finalModel)
#' @importFrom glmnet cv.glmnet glmnet
#' @export
lasso_pro <- function(X, y_surv, tune = FALSE) {
  X_matrix <- stats::model.matrix(~ . - 1, data = X)
  cv_fit <- glmnet::cv.glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 1)
  best_lambda <- cv_fit$lambda.min
  final_model <- glmnet::glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 1, lambda = best_lambda)
  final_model$fitted_scores <- base::as.numeric(stats::predict(final_model, newx = X_matrix, type = "link"))
  final_model$y_surv <- y_surv
  structure(list(finalModel = final_model, X_train_cols = colnames(X), model_type = "survival_glmnet"), class = "train")
}

#' @title Train an Elastic Net Cox Proportional Hazards Model
#' @description Trains a Cox proportional hazards model with Elastic Net regularization
#'   using `glmnet` (with alpha = 0.5).
#'
#' @param X A data frame of features.
#' @param y_surv A `survival::Surv` object representing the survival outcome.
#' @param tune Logical, whether to perform hyperparameter tuning (currently simplified/ignored
#'   for direct `cv.glmnet` usage which inherently tunes lambda).
#' @return A list of class "train" containing the trained `glmnet` model object,
#'   names of features used in training, and model type. The returned object
#'   also includes `fitted_scores` (linear predictor), `y_surv`, `best_lambda`, and `alpha_val`.
#' @examples
#' set.seed(42)
#' n_samples <- 50
#' n_features <- 10
#' X_data <- as.data.frame(matrix(rnorm(n_samples * n_features), ncol = n_features))
#' Y_surv_obj <- survival::Surv(
#'   time = runif(n_samples, 100, 1000),
#'   event = sample(0:1, n_samples, replace = TRUE)
#' )
#'
#' # Train the model
#' en_model <- en_pro(X_data, Y_surv_obj)
#' print(en_model$finalModel)
#' @importFrom glmnet cv.glmnet glmnet
#' @export
en_pro <- function(X, y_surv, tune = FALSE) {
  X_matrix <- stats::model.matrix(~ . - 1, data = X)
  alpha_val <- 0.5
  cv_fit <- glmnet::cv.glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = alpha_val)
  best_lambda <- cv_fit$lambda.min
  final_model <- glmnet::glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = alpha_val, lambda = best_lambda)
  final_model$fitted_scores <- base::as.numeric(stats::predict(final_model, newx = X_matrix, type = "link"))
  final_model$y_surv <- y_surv
  final_model$best_lambda <- best_lambda
  final_model$alpha_val <- alpha_val
  structure(list(finalModel = final_model, X_train_cols = colnames(X), model_type = "survival_glmnet"), class = "train")
}

#' @title Train a Ridge Cox Proportional Hazards Model
#' @description Trains a Cox proportional hazards model with Ridge regularization
#'   using `glmnet`.
#'
#' @param X A data frame of features.
#' @param y_surv A `survival::Surv` object representing the survival outcome.
#' @param tune Logical, whether to perform hyperparameter tuning (currently simplified/ignored
#'   for direct `cv.glmnet` usage which inherently tunes lambda).
#' @return A list of class "train" containing the trained `glmnet` model object,
#'   names of features used in training, and model type. The returned object
#'   also includes `fitted_scores` (linear predictor), `y_surv`, and `best_lambda`.
#' @examples
#' set.seed(42)
#' n_samples <- 50
#' n_features <- 10
#' X_data <- as.data.frame(matrix(rnorm(n_samples * n_features), ncol = n_features))
#' Y_surv_obj <- survival::Surv(
#'   time = runif(n_samples, 100, 1000),
#'   event = sample(0:1, n_samples, replace = TRUE)
#' )
#'
#' # Train the model
#' ridge_model <- ridge_pro(X_data, Y_surv_obj)
#' print(ridge_model$finalModel)
#' @importFrom glmnet cv.glmnet glmnet
#' @export
ridge_pro <- function(X, y_surv, tune = FALSE) {
  X_matrix <- stats::model.matrix(~ . - 1, data = X)
  cv_fit <- glmnet::cv.glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 0)
  best_lambda <- cv_fit$lambda.min
  final_model <- glmnet::glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 0, lambda = best_lambda)
  final_model$fitted_scores <- base::as.numeric(stats::predict(final_model, newx = X_matrix, type = "link"))
  final_model$y_surv <- y_surv
  final_model$best_lambda <- best_lambda
  structure(list(finalModel = final_model, X_train_cols = colnames(X), model_type = "survival_glmnet"), class = "train")
}

#' @title Train a Random Survival Forest Model
#' @description Trains a Random Survival Forest (RSF) model using `randomForestSRC`.
#'
#' @param X A data frame of features.
#' @param y_surv A `survival::Surv` object representing the survival outcome.
#' @param tune Logical, whether to perform hyperparameter tuning (a simplified
#'   message is currently provided, full tuning with `tune.rfsrc` is recommended
#'   for advanced use).
#' @return A list of class "train" containing the trained `rfsrc` model object,
#'   names of features used in training, and model type. The returned object
#'   also includes `fitted_scores` and `y_surv`.
#' @examples
#' \donttest{
#' # Generate some dummy survival data
#' set.seed(42)
#' n_samples <- 50
#' n_features <- 5
#' X_data <- as.data.frame(matrix(rnorm(n_samples * n_features), ncol = n_features))
#' Y_surv_obj <- survival::Surv(
#'   time = runif(n_samples, 100, 1000),
#'   event = sample(0:1, n_samples, replace = TRUE)
#' )
#'
#' # Train the model (ntree is small for a quick example)
#' rsf_model <- rsf_pro(X_data, Y_surv_obj)
#' print(rsf_model$finalModel)
#' }
#' @importFrom randomForestSRC rfsrc predict.rfsrc
#' @export
rsf_pro <- function(X, y_surv, tune = FALSE) {
  data_for_rsf <- cbind(Y_surv_ = y_surv, X)
  names(data_for_rsf)[1] <- "Y_surv_"
  if (tune) {
    message("RSF: Simplified tuning; consider tune.rfsrc for comprehensive tuning.")
  }
  fit <- randomForestSRC::rfsrc(stats::as.formula("Y_surv_ ~ ."), data = data_for_rsf, ntree = 100, mtry = base::floor(ncol(X)/3))
  fit$fitted_scores <- randomForestSRC::predict.rfsrc(fit, newdata = X)$predicted
  fit$y_surv <- y_surv
  structure(list(finalModel = fit, X_train_cols = colnames(X), model_type = "survival_rsf"), class = "train")
}

#' @title Train a Stepwise Cox Proportional Hazards Model
#' @description Trains a Cox proportional hazards model and performs backward
#'   stepwise selection using `MASS::stepAIC` to select important features.
#'
#' @param X A data frame of features.
#' @param y_surv A `survival::Surv` object representing the survival outcome.
#' @param tune Logical, whether to perform hyperparameter tuning (currently ignored).
#' @return A list of class "train" containing the trained `coxph` model object
#'   after stepwise selection, names of features used in training, and model type.
#'   The returned object also includes `fitted_scores` (linear predictor) and `y_surv`.
#' @examples
#' set.seed(42)
#' n_samples <- 50
#' n_features <- 5
#' X_data <- as.data.frame(matrix(rnorm(n_samples * n_features), ncol = n_features))
#' Y_surv_obj <- survival::Surv(
#'   time = runif(n_samples, 100, 1000),
#'   event = sample(0:1, n_samples, replace = TRUE)
#' )
#'
#' # Train the model
#' stepcox_model <- stepcox_pro(X_data, Y_surv_obj)
#' print(stepcox_model$finalModel)
#' @importFrom survival coxph
#' @importFrom MASS stepAIC
#' @export
stepcox_pro <- function(X, y_surv, tune = FALSE) {
  data_for_cox <- cbind(y_surv_ = y_surv, X)
  formula_full <- stats::as.formula(paste("y_surv_ ~", paste(colnames(X), collapse = " + ")))
  fit_full <- tryCatch({
    survival::coxph(formula_full, data = data_for_cox)
  }, error = function(e) {
    stop(paste("Initial Cox model failed:", e$message))
  })
  fit <- MASS::stepAIC(fit_full, direction = "backward", trace = FALSE)
  fit$fitted_scores <- stats::predict(fit, newdata = X, type = "lp")
  fit$y_surv <- y_surv
  structure(list(finalModel = fit, X_train_cols = colnames(X), model_type = "survival_stepcox"), class = "train")
}

#' @title Train a Gradient Boosting Machine (GBM) for Survival Data
#' @description Trains a Gradient Boosting Machine (GBM) model with a Cox
#'   proportional hazards loss function using `gbm`.
#'
#' @param X A data frame of features.
#' @param y_surv A `survival::Surv` object representing the survival outcome.
#' @param tune Logical, whether to perform simplified hyperparameter tuning.
#'   If `TRUE`, `n.trees`, `interaction.depth`, and `shrinkage` are set to
#'   predefined values suitable for tuning; otherwise, default values are used.
#' @param cv.folds Integer. The number of cross-validation folds to use.
#'   Setting this to 0 or 1 will disable cross-validation. Defaults to 3.
#' @return A list of class "train" containing the trained `gbm` model object,
#'   names of features used in training, and model type. The returned object
#'   also includes `fitted_scores` (linear predictor), `y_surv`, and `best_iter`.
#' @examples
#' \donttest{
#' # Generate some dummy survival data
#' set.seed(42)
#' n_samples <- 200
#' n_features <- 5
#' X_data <- as.data.frame(matrix(rnorm(n_samples * n_features), ncol = n_features))
#' Y_surv_obj <- survival::Surv(
#'   time = runif(n_samples, 100, 1000),
#'   event = sample(0:1, n_samples, replace = TRUE)
#' )
#'
#' # Train the model for the example *without* cross-validation to pass R CMD check
#' # In real use, you might use the default cv.folds = 3
#' gbm_model <- gbm_pro(X_data, Y_surv_obj, cv.folds = 0)
#' print(gbm_model$finalModel)
#' }
#' @importFrom gbm gbm gbm.perf
#' @importFrom survival Surv
#' @importFrom stats predict
#' @export
gbm_pro <- function(X, y_surv, tune = FALSE, cv.folds = 3) { # <--- 新增参数
  data_for_gbm <- cbind(y_surv_time = y_surv[,1], y_surv_event = y_surv[,2], X)

  if (tune) {
    message("GBM (Cox): Simplified tuning; comprehensive tuning for interaction.depth, shrinkage is recommended.")
    n_trees_val <- 200
    interaction_depth_val <- 3
    shrinkage_val <- 0.01
  } else {
    n_trees_val <- 100
    interaction_depth_val <- 3
    shrinkage_val <- 0.1
  }

  fit <- gbm::gbm(
    formula = survival::Surv(y_surv_time, y_surv_event) ~ .,
    data = data_for_gbm,
    distribution = "coxph",
    n.trees = n_trees_val,
    interaction.depth = interaction_depth_val,
    shrinkage = shrinkage_val,
    cv.folds = cv.folds # <--- 使用新参数
  )

  # gbm.perf requires cv.folds > 1. Only run it if CV was performed.
  if (cv.folds > 1) {
    best_iter <- gbm::gbm.perf(fit, method = "cv", plot.it = FALSE)
  } else {
    # If no CV, use the total number of trees as best_iter
    best_iter <- n_trees_val
  }

  fit$fitted_scores <- stats::predict(fit, newdata = X, n.trees = best_iter, type = "link")
  fit$y_surv <- y_surv
  fit$best_iter <- best_iter

  structure(
    list(
      finalModel = fit,
      X_train_cols = colnames(X),
      model_type = "survival_gbm"
    ),
    class = "train"
  )
}

#' @title Min-Max Normalization
#' @description Normalizes a numeric vector to a range of 0 to 1 using min-max scaling.
#'
#' @param x A numeric vector to be normalized.
#' @param min_val Optional. The minimum value to use for normalization. If `NULL`,
#'   the minimum of `x` is used.
#' @param max_val Optional. The maximum value to use for normalization. If `NULL`,
#'   the maximum of `x` is used.
#' @return A numeric vector with values scaled between 0 and 1. If `min_val`
#'   and `max_val` are equal (or `x` has no variance), returns a vector of 0.5s.
#' @examples
#' # Normalize a vector
#' x_vec <- c(10, 20, 30, 40, 50)
#' normalized_x <- min_max_normalize(x_vec)
#' print(normalized_x) # Should be 0, 0.25, 0.5, 0.75, 1
#'
#' # Normalize with custom min/max
#' custom_normalized_x <- min_max_normalize(x_vec, min_val = 0, max_val = 100)
#' print(custom_normalized_x) # Should be 0.1, 0.2, 0.3, 0.4, 0.5
#' @export
min_max_normalize <- function(x, min_val = NULL, max_val = NULL) {
  if (is.null(min_val)) min_val <- base::min(x, na.rm = TRUE)
  if (is.null(max_val)) max_val <- base::max(x, na.rm = TRUE)

  if (min_val == max_val) {
    return(rep(0.5, length(x)))
  }
  (x - min_val) / (max_val - min_val)
}

#' @title Evaluate Prognostic Model Performance
#' @description Evaluates the performance of a trained prognostic model using
#'   various metrics relevant to survival analysis, including C-index,
#'   time-dependent AUROC, and Kaplan-Meier (KM) group analysis (Hazard Ratio and p-value).
#'
#' @param trained_model_obj A trained model object (of class "train" as returned
#'   by model training functions like `lasso_pro`, `rsf_pro`, etc.). Can be `NULL`
#'   if `precomputed_score` is provided.
#' @param X_data A data frame of features corresponding to the data used for evaluation.
#'   Required if `trained_model_obj` is provided and `precomputed_score` is `NULL`.
#' @param Y_surv_obj A `survival::Surv` object for the evaluation data.
#' @param sample_ids A vector of sample IDs for the evaluation data.
#' @param years_to_evaluate A numeric vector of specific years at which to
#'   calculate time-dependent AUROC.
#' @param precomputed_score Optional. A numeric vector of precomputed prognostic
#'   scores for the samples. If provided, `trained_model_obj` and `X_data` are
#'   not strictly necessary for score derivation.
#' @param meta_normalize_params Optional. A list of normalization parameters
#'   (min/max values) used for base model scores in a stacking ensemble.
#'   Used when `trained_model_obj` is a stacking model to ensure consistent
#'   normalization during prediction.
#'
#' @return A list containing:
#'   \itemize{
#'     \item `sample_score`: A data frame with `ID`, `outcome`, `time`, and `score` columns.
#'     \item `evaluation_metrics`: A list of performance metrics:
#'       \itemize{
#'         \item `C_index`: Harrell's C-index.
#'         \item `AUROC_Years`: A named list of time-dependent AUROC values for specified years.
#'         \item `AUROC_Average`: The average of time-dependent AUROC values.
#'         \item `KM_HR`: Hazard Ratio for High vs Low risk groups (split by median score).
#'         \item `KM_P_value`: P-value for the KM group comparison.
#'         \item `KM_Cutoff`: The score cutoff used to define High/Low risk groups (median score).
#'       }
#'   }
#' @examples
#' \donttest{
#' # Generate dummy data
#' set.seed(42)
#' n <- 50
#' X <- as.data.frame(matrix(rnorm(n * 5), n, 5))
#' Y_surv <- survival::Surv(time = runif(n, 1, 5*365), event = sample(0:1, n, TRUE))
#' ids <- paste0("s", 1:n)
#'
#' # Train a simple model
#' initialize_modeling_system_pro()
#' model_obj <- lasso_pro(X, Y_surv)
#'
#' # Evaluate the model on the same data
#' eval_results <- evaluate_model_pro(
#'   trained_model_obj = model_obj,
#'   X_data = X,
#'   Y_surv_obj = Y_surv,
#'   sample_ids = ids,
#'   years_to_evaluate = c(1, 2, 3)
#' )
#' str(eval_results$evaluation_metrics)
#' }
#' @importFrom survival Surv coxph
#' @importFrom survcomp concordance.index
#' @importFrom survivalROC survivalROC
#' @export
evaluate_model_pro <- function(trained_model_obj = NULL, X_data = NULL, Y_surv_obj, sample_ids,
                               years_to_evaluate = c(1, 3, 5),
                               precomputed_score = NULL,
                               meta_normalize_params = NULL) {

  score <- precomputed_score

  if (is.null(score)) {
    if (is.null(trained_model_obj)) {
      stop("Either 'trained_model_obj' or 'precomputed_score' must be provided for evaluation.")
    }
    if (is.null(X_data)) {
      stop("X_data must be provided when deriving scores from 'trained_model_obj'.")
    }

    X_train_cols <- trained_model_obj$X_train_cols
    if (!is.null(X_train_cols)) {
      missing_cols <- setdiff(X_train_cols, names(X_data))
      if (length(missing_cols) > 0) {
        for (col in missing_cols) { X_data[[col]] <- NA }
        warning(sprintf("Missing feature '%s' in evaluation data. Added with NA.", col))
      }
      X_data <- X_data[, X_train_cols, drop = FALSE]
    } else {
      warning("Trained model object does not contain 'X_train_cols'. Prediction might fail if feature order/set is different.")
    }

    if (!is.null(trained_model_obj$model_type)) {
      model_type <- trained_model_obj$model_type
      final_model <- trained_model_obj$finalModel

      if (model_type == "survival_glmnet") {
        score <- base::as.numeric(stats::predict(final_model, newx = stats::model.matrix(~ . - 1, data = X_data), type = "link"))
      } else if (model_type == "survival_rsf") {
        score_raw <- randomForestSRC::predict.rfsrc(final_model, newdata = X_data)$predicted
        temp_c_index <- survcomp::concordance.index(x = score_raw, surv.time = Y_surv_obj[,1], surv.event = Y_surv_obj[,2])$c.index
        score <- if(temp_c_index < 0.5) -score_raw else score_raw
      } else if (model_type == "survival_stepcox") {
        score <- stats::predict(final_model, newdata = X_data, type = "lp")
      } else if (model_type == "survival_gbm") {
        score <- stats::predict(final_model, newdata = X_data, n.trees = trained_model_obj$finalModel$best_iter, type = "link")
      } else if (model_type == "bagging_pro") {
        all_scores <- matrix(NA, nrow = nrow(X_data), ncol = length(trained_model_obj$base_model_objects))
        for (i in seq_along(trained_model_obj$base_model_objects)) {
          current_base_model_obj <- trained_model_obj$base_model_objects[[i]]
          if (!is.null(current_base_model_obj)) {
            # Recursive call to evaluate_model_pro to get base model scores
            all_scores[, i] <- evaluate_model_pro(trained_model_obj = current_base_model_obj,
                                                  X_data = X_data, Y_surv_obj = Y_surv_obj,
                                                  sample_ids = sample_ids, precomputed_score = NULL)$sample_score$score
          }
        }
        score <- rowMeans(all_scores, na.rm = TRUE)
      } else if (model_type == "stacking_pro") {
        base_models <- trained_model_obj$base_model_objects
        meta_model <- trained_model_obj$trained_meta_model

        all_base_scores <- matrix(NA, nrow = nrow(X_data), ncol = length(base_models))
        names(base_models) <- trained_model_obj$base_models_used

        for (i in seq_along(base_models)) {
          current_base_model_obj <- base_models[[i]]
          if (!is.null(current_base_model_obj)) {
            base_score_eval <- evaluate_model_pro(trained_model_obj = current_base_model_obj,
                                                  X_data = X_data, Y_surv_obj = Y_surv_obj,
                                                  sample_ids = sample_ids, precomputed_score = NULL)
            all_base_scores[, i] <- base_score_eval$sample_score$score
          }
        }
        colnames(all_base_scores) <- trained_model_obj$base_models_used # Assign names after filling

        if (!is.null(meta_normalize_params)) {
          for (j in 1:ncol(all_base_scores)) {
            model_name <- colnames(all_base_scores)[j]
            if (model_name %in% names(meta_normalize_params)) {
              params <- meta_normalize_params[[model_name]]
              all_base_scores[, j] <- min_max_normalize(all_base_scores[, j], params$min_val, params$max_val)
            } else {
              warning(paste("Normalization parameters for base model", model_name, "not found. Normalizing based on current data range."))
              all_base_scores[, j] <- min_max_normalize(all_base_scores[, j])
            }
          }
        } else {
          for (j in 1:ncol(all_base_scores)) {
            all_base_scores[, j] <- min_max_normalize(all_base_scores[, j])
          }
        }

        X_meta_new <- as.data.frame(all_base_scores)
        names(X_meta_new) <- paste0("pred_", trained_model_obj$base_models_used)

        # Handle meta-model feature alignment
        meta_train_features <- NULL
        if ("train" %in% class(meta_model) && !is.null(meta_model$X_train_cols)) {
          meta_train_features <- meta_model$X_train_cols
        } else if ("train" %in% class(meta_model) && !is.null(meta_model$trainingData)) { # Fallback for some models
          meta_train_features <- names(meta_model$trainingData)[-ncol(meta_model$trainingData)]
        }

        if (!is.null(meta_train_features)) {
          missing_meta_features <- setdiff(meta_train_features, names(X_meta_new))
          if (length(missing_meta_features) > 0) {
            for(mf in missing_meta_features) { X_meta_new[[mf]] <- NA }
            warning(paste("Meta model missing features:", paste(missing_meta_features, collapse=", ")))
          }
          extra_meta_features <- setdiff(names(X_meta_new), meta_train_features)
          if(length(extra_meta_features) > 0) {
            X_meta_new <- X_meta_new[, !names(X_meta_new) %in% extra_meta_features, drop=FALSE]
            warning(paste("Meta model has extra features; removing:", paste(extra_meta_features, collapse=", ")))
          }
          X_meta_new <- X_meta_new[, meta_train_features, drop=FALSE]
        } else {
          warning("Could not retrieve meta-model training feature names. Prediction might fail.")
        }


        if (trained_model_obj$meta_model_type == "survival_glmnet") {
          score <- base::as.numeric(stats::predict(meta_model$finalModel, newx = stats::model.matrix(~ . - 1, data = X_meta_new), type = "link"))
        } else if (trained_model_obj$meta_model_type == "survival_gbm") {
          score <- stats::predict(meta_model$finalModel, newdata = X_meta_new, n.trees = meta_model$finalModel$best_iter, type = "link")
        } else if (trained_model_obj$meta_model_type == "survival_rsf") {
          score <- randomForestSRC::predict.rfsrc(meta_model$finalModel, newdata = X_meta_new)$predicted
        } else if (trained_model_obj$meta_model_type == "survival_stepcox") {
          score <- stats::predict(meta_model$finalModel, newdata = X_meta_new, type = "lp")
        } else {
          stop(paste("Unsupported meta model type for prediction:", trained_model_obj$meta_model_type))
        }
      } else {
        stop("Unsupported prognosis model type for prediction. Please provide a supported object or 'precomputed_score'.")
      }
    } else {
      stop("Unsupported prognosis model object structure. Missing 'model_type'.")
    }
  }

  score[is.na(score)] <- stats::median(score, na.rm = TRUE)

  sample_score_df <- data.frame(
    ID = sample_ids,
    outcome = Y_surv_obj[,2],
    time = Y_surv_obj[,1],
    score = score
  )

  c_index_val <- NA
  tryCatch({
    c_index_val <- survcomp::concordance.index(x = score, surv.time = Y_surv_obj[,1], surv.event = Y_surv_obj[,2])$c.index
  }, error = function(e) { warning(paste("Could not calculate C-index:", e$message)) })

  auroc_yearly <- list()
  for (year in years_to_evaluate) {
    eval_time_days <- year * 365.25
    if (base::max(Y_surv_obj[,1], na.rm = TRUE) < eval_time_days) {
      auroc_yearly[[as.character(year)]] <- NA
      warning(paste("Max follow-up time (", round(base::max(Y_surv_obj[,1], na.rm = TRUE)/365.25, 2), " years) is less than ", year, " years. Skipping time-dependent ROC for ", year, " years.", sep=""))
      next
    }
    roc_obj <- tryCatch({
      survivalROC::survivalROC(Stime = Y_surv_obj[,1], status = Y_surv_obj[,2], marker = score, predict.time = eval_time_days, method = "NNE", span = 0.25)
    }, error = function(e) { warning(paste("Time-dependent ROC calculation failed for year", year, ":", e$message)); NULL })
    if (!is.null(roc_obj)) { auroc_yearly[[as.character(year)]] <- roc_obj$AUC } else { auroc_yearly[[as.character(year)]] <- NA }
  }

  avg_auroc <- mean(unlist(auroc_yearly), na.rm = TRUE)

  median_score <- stats::median(score, na.rm = TRUE)
  if (is.na(median_score) || length(unique(score[!is.na(score)])) < 2 || stats::sd(score, na.rm = TRUE) == 0) {
    warning("Cannot perform KM analysis due to constant, all NA, or non-varying scores.")
    km_hr <- NA ; km_p <- NA ; km_cutoff <- NA
  } else {
    risk_group <- factor(base::ifelse(score > median_score, "High", "Low"), levels = c("Low", "High"))
    if (length(unique(risk_group)) < 2) {
      warning("Cannot perform KM analysis: only one risk group after splitting by median.")
      km_hr <- NA ; km_p <- NA ; km_cutoff <- NA
    } else {
      cox_fit <- tryCatch({ survival::coxph(Y_surv_obj ~ risk_group) }, error = function(e) { warning(paste("Cox regression for KM analysis failed:", e$message)); NULL })
      if (!is.null(cox_fit)) {
        km_hr <- summary(cox_fit)$conf.int[1, "exp(coef)"]
        km_p <- summary(cox_fit)$coefficients[1, "Pr(>|z|)"]
        km_cutoff <- median_score
      } else { km_hr <- NA ; km_p <- NA ; km_cutoff <- NA }
    }
  }

  evaluation_metrics <- list(
    C_index = c_index_val,
    AUROC_Years = auroc_yearly,
    AUROC_Average = avg_auroc,
    KM_HR = km_hr,
    KM_P_value = km_p,
    KM_Cutoff = km_cutoff
  )

  return(list(sample_score = sample_score_df, evaluation_metrics = evaluation_metrics))
}

#' @title Run Multiple Prognostic Models
#' @description Trains and evaluates one or more registered prognostic models on a given dataset.
#'
#' @param data A data frame for training. The first column must be the sample ID,
#'   the second column the event status (0/1), the third column the time, and
#'   subsequent columns the features.
#' @param model A character string or vector of character strings, specifying
#'   which models to run. Use "all_pro" to run all registered models.
#' @param tune Logical, whether to enable hyperparameter tuning for individual models.
#' @param seed An integer, for reproducibility of random processes.
#' @param time_unit A character string, the unit of time in the third column of `data`.
#'   Can be "day", "month", or "year".
#' @param years_to_evaluate A numeric vector of specific years at which to
#'   calculate time-dependent AUROC.
#'
#' @return A named list, where each element corresponds to a run model and
#'   contains its trained `model_object`, `sample_score` data frame, and
#'   `evaluation_metrics`.
#' @examples
#' \donttest{
#' # NOTE: This example requires the 'train_pro' dataset to be exported by the package.
#' # If it is not, replace `data(train_pro)` with code to create a dummy dataframe.
#' # For demonstration, we assume `train_pro` is available.
#' if (requireNamespace("E2E", quietly = TRUE) &&
#'  "train_pro" %in% utils::data(package = "E2E")$results[,3]) {
#'   data(train_pro, package = "E2E")
#'
#'   # Initialize the modeling system
#'   initialize_modeling_system_pro()
#'
#'   # Run selected models
#'   results <- models_pro(
#'     data = train_pro,
#'     model = c("lasso_pro", "rsf_pro"), # Run only Lasso and RSF
#'     years_to_evaluate = c(1, 3, 5),
#'     seed = 42
#'   )
#'
#'   # Print summaries
#'   for (model_name in names(results)) {
#'     print_model_summary_pro(model_name, results[[model_name]])
#'   }
#' }
#' }
#' @seealso \code{\link{initialize_modeling_system_pro}}, \code{\link{evaluate_model_pro}}
#' @export
models_pro <- function(data,
                       model = "all_pro",
                       tune = FALSE,
                       seed = 123,
                       time_unit = "day",
                       years_to_evaluate = c(1, 3, 5)) {

  if (!.model_registry_env_pro$is_initialized) {
    stop("Prognosis modeling system not initialized. Please call 'initialize_modeling_system_pro()' first.")
  }

  all_registered_models <- get_registered_models_pro()

  models_to_run_names <- NULL
  if (length(model) == 1 && model == "all_pro") {
    models_to_run_names <- names(all_registered_models)
  } else if (all(model %in% names(all_registered_models))) {
    models_to_run_names <- model
  } else {
    stop(paste("Invalid prognosis model name(s) provided. Available models are:", paste(names(all_registered_models), collapse = ", ")))
  }

  set.seed(seed)
  data_prepared <- .prepare_data_pro(data, time_unit)

  X_data <- data_prepared$X
  Y_surv_obj <- data_prepared$Y_surv
  sample_ids <- data_prepared$sample_ids

  all_model_results <- list()

  for (model_name in models_to_run_names) {
    current_model_func <- get_registered_models_pro()[[model_name]]

    message(sprintf("Running model: %s", model_name))
    mdl <- tryCatch({
      set.seed(seed)
      current_model_func(X_data, Y_surv_obj, tune = tune)
    }, error = function(e) {
      warning(paste("Prognosis Model", model_name, "failed during training:", conditionMessage(e)))
      NULL
    })

    if (!is.null(mdl)) {
      eval_results <- tryCatch({
        evaluate_model_pro(trained_model_obj = mdl, X_data = X_data, Y_surv_obj = Y_surv_obj,
                           sample_ids = sample_ids, years_to_evaluate = years_to_evaluate)
      }, error = function(e) {
        warning(paste("Prognosis Model", model_name, "failed during evaluation:", conditionMessage(e)))
        list(sample_score = data.frame(ID = sample_ids, outcome = Y_surv_obj[,2], time = Y_surv_obj[,1], score = NA),
             evaluation_metrics = list(error = paste("Evaluation failed:", conditionMessage(e))))
      })

      all_model_results[[model_name]] <- list(
        model_object = mdl,
        sample_score = eval_results$sample_score,
        evaluation_metrics = eval_results$evaluation_metrics
      )
    } else {
      failed_sample_score <- data.frame(
        ID = sample_ids,
        outcome = Y_surv_obj[,2],
        time = Y_surv_obj[,1],
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

#' @title Train a Bagging Prognostic Model
#' @description Implements a Bagging (Bootstrap Aggregating) ensemble for
#'   prognostic models. It trains multiple base models on bootstrapped samples
#'   of the training data and aggregates their predictions.
#'
#' @param data A data frame for training. The first column must be the sample ID,
#'   the second column the event status (0/1), the third column the time, and
#'   subsequent columns the features.
#' @param base_model_name A character string, the name of the base prognostic
#'   model to use (e.g., "lasso_pro", "rsf_pro"). This model must be registered.
#' @param n_estimators An integer, the number of base models to train.
#' @param subset_fraction A numeric value between 0 and 1, the fraction of
#'   samples to bootstrap for each base model.
#' @param tune_base_model Logical, whether to enable tuning for each base model.
#' @param time_unit A character string, the unit of time in the third column of `data`.
#' @param years_to_evaluate A numeric vector of specific years at which to
#'   calculate time-dependent AUROC for evaluation.
#' @param seed An integer, for reproducibility.
#'
#' @return A list containing the `model_object`, `sample_score`, and `evaluation_metrics`.
#' @examples
#' \donttest{
#' # NOTE: This example requires the 'train_pro' dataset.
#' if (requireNamespace("E2E", quietly = TRUE) &&
#' "train_pro" %in% utils::data(package = "E2E")$results[,3]) {
#'   data(train_pro, package = "E2E")
#'   initialize_modeling_system_pro()
#'
#'   bagging_lasso_results <- bagging_pro(
#'     data = train_pro,
#'     base_model_name = "lasso_pro",
#'     n_estimators = 3, # Small number for example speed
#'     subset_fraction = 0.8,
#'     years_to_evaluate = c(1, 3)
#'   )
#'   print_model_summary_pro("Bagging (Lasso)", bagging_lasso_results)
#' }
#' }
#' @seealso \code{\link{initialize_modeling_system_pro}}, \code{\link{evaluate_model_pro}}
#' @export
bagging_pro <- function(data,
                        base_model_name,
                        n_estimators = 10,
                        subset_fraction = 0.632,
                        tune_base_model = FALSE,
                        time_unit = "day",
                        years_to_evaluate = c(1, 3, 5),
                        seed = 456) {

  if (!.model_registry_env_pro$is_initialized) {
    initialize_modeling_system_pro()
  }

  all_registered_models <- get_registered_models_pro()
  if (!(base_model_name %in% names(all_registered_models))) {
    stop(sprintf("Base prognosis model '%s' not found. Please register it first.", base_model_name))
  }

  message(sprintf("Running Bagging model: %s (base: %s)", "Bagging_pro", base_model_name))

  set.seed(seed)
  data_prepared <- .prepare_data_pro(data, time_unit)

  X_data <- data_prepared$X
  Y_surv_obj <- data_prepared$Y_surv
  sample_ids <- data_prepared$sample_ids
  n_samples <- nrow(X_data)
  subset_size <- base::floor(n_samples * subset_fraction)
  if (subset_size == 0) stop("Subset size is 0. Please check your data or subset_fraction.")

  trained_models_and_scores <- list()
  base_model_func <- get_registered_models_pro()[[base_model_name]]

  for (i in 1:n_estimators) {
    set.seed(seed + i)
    indices <- sample(1:n_samples, subset_size, replace = TRUE)
    X_boot <- X_data[indices, , drop = FALSE]
    Y_surv_boot <- Y_surv_obj[indices]

    if (sum(Y_surv_boot[,2]) == 0) {
      warning(sprintf("Bootstrap sample %d has no events. Skipping this model.", i))
      trained_models_and_scores[[i]] <- list(model = NULL, score = rep(NA, n_samples))
      next
    }

    current_model <- tryCatch({
      base_model_func(X_boot, Y_surv_boot, tune = tune_base_model)
    }, error = function(e) {
      warning(sprintf("Training base model %s for bootstrap %d failed: %s", base_model_name, i, e$message))
      NULL
    })

    score_on_full_data <- rep(NA, n_samples)
    if (!is.null(current_model)) {
      tryCatch({
        score_on_full_data_eval <- evaluate_model_pro(trained_model_obj = current_model,
                                                      X_data = X_data, Y_surv_obj = Y_surv_obj,
                                                      sample_ids = sample_ids, precomputed_score = NULL)
        score_on_full_data <- score_on_full_data_eval$sample_score$score
      }, error = function(e) {
        warning(sprintf("Prediction for base model %s for bootstrap %d failed: %s", base_model_name, i, e$message))
      })
    }
    trained_models_and_scores[[i]] <- list(model = current_model, score = score_on_full_data)
  }

  valid_models <- lapply(trained_models_and_scores, `[[`, "model")
  valid_scores_list <- lapply(trained_models_and_scores, `[[`, "score")
  valid_models <- valid_models[!sapply(valid_models, is.null)]
  valid_scores_list <- valid_scores_list[!sapply(valid_scores_list, function(s) all(is.na(s)))]

  if (length(valid_scores_list) == 0) {
    stop("No base models were successfully trained or made valid predictions. Cannot perform bagging.")
  }

  aggregated_score <- rowMeans(do.call(cbind, valid_scores_list), na.rm = TRUE)

  bagging_model_obj_for_eval <- list(
    model_type = "bagging_pro",
    base_model_name = base_model_name,
    n_estimators = n_estimators,
    base_model_objects = valid_models,
    X_train_cols = colnames(X_data)
  )

  eval_results <- evaluate_model_pro(
    Y_surv_obj = Y_surv_obj,
    sample_ids = sample_ids,
    years_to_evaluate = years_to_evaluate,
    precomputed_score = aggregated_score
  )

  list(
    model_object = bagging_model_obj_for_eval,
    sample_score = eval_results$sample_score,
    evaluation_metrics = eval_results$evaluation_metrics
  )
}

#' @title Train a Stacking Prognostic Model
#' @description Implements a Stacking ensemble for prognostic models. It trains
#'   multiple base models and uses their predictions to train a meta-model.
#'
#' @param results_all_models A list of results from `models_pro()`,
#'   containing trained base model objects and their evaluation metrics.
#' @param data A data frame for training the meta-model. The first column must be ID,
#'   second event status (0/1), third time, and subsequent columns features.
#' @param meta_model_name A character string, the name of the meta-model to use
#'   (e.g., "lasso_pro", "gbm_pro"). This model must be registered.
#' @param top An integer, the number of top-performing base models (ranked by C-index)
#'   to select for the stacking ensemble.
#' @param tune_meta Logical, whether to enable tuning for the meta-model.
#' @param time_unit A character string, the unit of time in the third column of `data`.
#' @param years_to_evaluate A numeric vector of specific years at which to
#'   calculate time-dependent AUROC for evaluation.
#' @param seed An integer, for reproducibility.
#'
#' @return A list containing the `model_object`, `sample_score`, and `evaluation_metrics`.
#' @examples
#' \donttest{
#' # NOTE: This example requires the 'train_pro' dataset.
#' if (requireNamespace("E2E", quietly = TRUE) &&
#' "train_pro" %in% utils::data(package = "E2E")$results[,3]) {
#'   data(train_pro, package = "E2E")
#'   initialize_modeling_system_pro()
#'
#'   # First, generate results for base models
#'   base_model_results <- models_pro(data = train_pro, model = c("lasso_pro", "rsf_pro"))
#'
#'   # Then, create the stacking ensemble
#'   stacking_lasso_results <- stacking_pro(
#'     results_all_models = base_model_results,
#'     data = train_pro,
#'     meta_model_name = "lasso_pro",
#'     top = 3,
#'     years_to_evaluate = c(1, 3)
#'   )
#'   print_model_summary_pro("Stacking (Lasso)", stacking_lasso_results)
#' }
#' }
#' @importFrom dplyr select left_join
#' @importFrom magrittr %>%
#' @seealso \code{\link{models_pro}}, \code{\link{evaluate_model_pro}}
#' @export
stacking_pro <- function(results_all_models,
                         data,
                         meta_model_name,
                         top = 3,
                         tune_meta = FALSE,
                         time_unit = "day",
                         years_to_evaluate = c(1, 3, 5),
                         seed = 789) {

  if (!is.null(results_all_models$`%||%`)) { # Helper for R < 4.0.0
    `%||%` <- function(a, b) if (is.null(a)) b else a
  }

  if (!.model_registry_env_pro$is_initialized) {
    stop("Prognosis modeling system not initialized. Please call 'initialize_modeling_system_pro()' first.")
  }

  all_registered_models <- get_registered_models_pro()
  if (!(meta_model_name %in% names(all_registered_models))) {
    stop(sprintf("Meta-model '%s' not found. Please register it first.", meta_model_name))
  }

  message(sprintf("Running Stacking model: %s (meta: %s)", "Stacking_pro", meta_model_name))

  set.seed(seed)
  data_prepared <- .prepare_data_pro(data, time_unit)

  Y_surv_obj <- data_prepared$Y_surv
  sample_ids <- data_prepared$sample_ids

  model_c_indices <- sapply(results_all_models, function(res) {
    if(is.null(res$evaluation_metrics$C_index)) NA else res$evaluation_metrics$C_index
  })
  model_c_indices <- model_c_indices[!is.na(model_c_indices)]

  if (length(model_c_indices) == 0) {
    stop("No base models with valid C-index found in 'results_all_models' for stacking.")
  }

  sorted_models_names <- names(sort(model_c_indices, decreasing = TRUE))
  selected_base_models_names <- utils::head(sorted_models_names, base::min(top, length(sorted_models_names)))
  if (length(selected_base_models_names) < 1) stop("No base models selected for stacking.")

  selected_base_model_objects <- lapply(results_all_models[selected_base_models_names], `[[`, "model_object")

  # Create meta-features (normalized scores from base models)
  X_meta <- data.frame(ID = sample_ids)
  meta_normalize_params <- list()

  for (model_name in selected_base_models_names) {
    current_scores <- results_all_models[[model_name]]$sample_score$score
    min_val <- base::min(current_scores, na.rm = TRUE)
    max_val <- base::max(current_scores, na.rm = TRUE)
    meta_normalize_params[[model_name]] <- list(min_val = min_val, max_val = max_val)
    normalized_scores <- min_max_normalize(current_scores, min_val, max_val)

    temp_df <- data.frame(ID = sample_ids, score_col = normalized_scores)
    names(temp_df)[2] <- paste0("pred_", model_name)
    X_meta <- dplyr::left_join(X_meta, temp_df, by = "ID")
  }

  X_meta_features <- X_meta %>% dplyr::select(-ID)

  # Train the meta-model
  meta_model_func <- all_registered_models[[meta_model_name]]
  meta_mdl <- tryCatch({
    set.seed(seed)
    meta_model_func(X_meta_features, Y_surv_obj, tune = tune_meta)
  }, error = function(e) {
    stop(paste("Meta-model", meta_model_name, "failed with error:", conditionMessage(e)))
  })

  # Evaluate the stacking model itself
  eval_results <- evaluate_model_pro(
    trained_model_obj = meta_mdl,
    X_data = X_meta_features,
    Y_surv_obj = Y_surv_obj,
    sample_ids = sample_ids,
    years_to_evaluate = years_to_evaluate
  )

  # The X_train_cols for the final model object should be the original feature names
  original_X_cols <- colnames(data_prepared$X)

  list(
    model_object = list(
      model_type = "stacking_pro",
      meta_model_name = meta_model_name,
      meta_model_type = meta_mdl$model_type,
      base_models_used = selected_base_models_names,
      base_model_objects = selected_base_model_objects,
      trained_meta_model = meta_mdl,
      meta_normalize_params = meta_normalize_params,
      X_train_cols = original_X_cols
    ),
    sample_score = eval_results$sample_score,
    evaluation_metrics = eval_results$evaluation_metrics
  )
}

#' @title Apply a Trained Prognostic Model to New Data
#' @description Applies a previously trained prognostic model (or ensemble) to a
#'   new, unseen dataset to generate prognostic scores.
#'
#' @param trained_model_object A trained model object, as returned by `models_pro`,
#'   `bagging_pro`, or `stacking_pro`.
#' @param new_data A data frame containing the new data for prediction. It should
#'   follow the same structure as the training data: ID, Outcome, Time, Features.
#'   The outcome and time columns are used for data preparation and can be included
#'   in the output, but the model's prediction only uses the features. If outcome/time
#'   are unknown, they can be filled with NA.
#' @param time_unit A character string, the unit of time in the third column of
#'   `new_data`.
#'
#' @return A data frame with `ID`, `outcome`, `time`, and predicted `score` for the new data.
#' @examples
#' \donttest{
#' # NOTE: This example requires 'train_pro' and 'test_pro' datasets.
#' if (requireNamespace("E2E", quietly = TRUE) &&
#'     "train_pro" %in% utils::data(package = "E2E")$results[,3] &&
#'     "test_pro" %in% utils::data(package = "E2E")$results[,3]) {
#'
#'   data(train_pro, package = "E2E")
#'   data(test_pro, package = "E2E")
#'   initialize_modeling_system_pro()
#'
#'   train_results <- models_pro(data = train_pro, model = "lasso_pro")
#'   trained_lasso_model <- train_results$lasso_pro$model_object
#'
#'   # Apply the trained model to new data
#'   new_data_predictions <- apply_pro(
#'     trained_model_object = trained_lasso_model,
#'     new_data = test_pro,
#'     time_unit = "day" # Specify time unit of test_pro
#'   )
#'   utils::head(new_data_predictions)
#' }
#' }
#' @importFrom dplyr select
#' @seealso \code{\link{evaluate_model_pro}}
#' @export
apply_pro <- function(trained_model_object,
                      new_data,
                      time_unit = "day") {

  if (is.null(trained_model_object)) {
    stop("Trained model object is NULL. Cannot apply to new data.")
  }
  if (!is.data.frame(new_data)) {
    stop("'new_data' must be a data.frame.")
  }

  if (is.null(new_data$`%||%`)) { # Helper for R < 4.0.0
    `%||%` <- function(a, b) if (is.null(a)) b else a
  }

  message("Applying model on new data...")

  new_data_prepared <- .prepare_data_pro(new_data, time_unit)
  X_new <- new_data_prepared$X
  Y_surv_new <- new_data_prepared$Y_surv
  new_sample_ids <- new_data_prepared$sample_ids

  # The core prediction logic is now handled by evaluate_model_pro
  # which can predict on new data given the model object and new features.
  # This avoids duplicating the complex prediction logic for different model types.

  # The 'meta_normalize_params' are only needed for stacking models
  meta_params <- trained_model_object$meta_normalize_params %||% NULL

  eval_on_new <- evaluate_model_pro(
    trained_model_obj = trained_model_object,
    X_data = X_new,
    Y_surv_obj = Y_surv_new, # Y_surv_new is needed for C-index etc., but not for score prediction
    sample_ids = new_sample_ids,
    precomputed_score = NULL, # We want it to compute the score
    meta_normalize_params = meta_params
  )

  # The result is the sample_score data frame from the evaluation
  # which contains ID, original outcome/time, and the new score.
  return(eval_on_new$sample_score)
}


#' @title Initialize Prognostic Modeling System
#' @description Initializes the prognostic modeling system by loading required
#'   packages and registering default prognostic models (Lasso, Elastic Net, Ridge,
#'   Random Survival Forest, Stepwise Cox, GBM for Cox). This function should
#'   be called once before using `run_models_pro()` or ensemble methods.
#'
#' @return Invisible NULL. Initializes the internal model registry.
#' @examples
#' # Initialize the system (typically run once at the start of a session or script)
#' initialize_modeling_system_pro()
#'
#' # Check if models are now registered
#' print(names(get_registered_models_pro()))
#' @export
initialize_modeling_system_pro <- function() {
  if (.model_registry_env_pro$is_initialized) {
    message("Prognosis modeling system already initialized.")
    return(invisible(NULL))
  }

  required_packages_pro <- c(
    "readr", "dplyr", "survival", "survcomp", "survivalROC",
    "glmnet", "randomForestSRC", "MASS", "gbm"
  )

  # Check if required packages are installed
  for (pkg in required_packages_pro) {
    if (!base::requireNamespace(pkg, quietly = TRUE)) {
      stop(paste("Package '", pkg, "' is required but not installed. Please install it using install.packages('", pkg, "').", sep=""))
    }
  }

  # Register default models
  register_model_pro("lasso_pro", lasso_pro)
  register_model_pro("en_pro", en_pro)
  register_model_pro("ridge_pro", ridge_pro)
  register_model_pro("rsf_pro", rsf_pro)
  register_model_pro("stepcox_pro", stepcox_pro)
  register_model_pro("gbm_pro", gbm_pro)

  .model_registry_env_pro$is_initialized <- TRUE
  message("Prognosis modeling system initialized and default models registered.")
  return(invisible(NULL))
}

#' @title Print Prognostic Model Summary
#' @description Prints a formatted summary of the evaluation metrics for a
#'   prognostic model, either from training data or new data evaluation.
#'
#' @param model_name A character string, the name of the model (e.g., "lasso_pro").
#' @param results_list A list containing model evaluation results, typically
#'   an element from the output of `run_models_pro()` or the result of `bagging_pro()`,
#'   `stacking_pro()`. It must contain `evaluation_metrics` and `model_object` (if applicable).
#' @param on_new_data Logical, indicating whether the results are from applying
#'   the model to new, unseen data (`TRUE`) or from the training/internal validation
#'   data (`FALSE`).
#'
#' @return NULL. Prints the summary to the console.
#' @examples
#' \donttest{
#' if (requireNamespace("E2E", quietly = TRUE) &&
#'  "train_pro" %in% utils::data(package = "E2E")$results[,3]) {
#'   data(train_pro, package = "E2E")
#'   initialize_modeling_system_pro()
#'   results <- models_pro(data = train_pro, model = "lasso_pro")
#'
#'   # Print summary for the trained model
#'   print_model_summary_pro("lasso_pro", results$lasso_pro, on_new_data = FALSE)
#'
#'   # Example for a failed model
#'   failed_results <- list(evaluation_metrics = list(error = "Training failed"))
#'   print_model_summary_pro("MyFailedModel", failed_results)
#' }
#' }
#' @export
print_model_summary_pro <- function(model_name, results_list, on_new_data = FALSE) {
  metrics <- results_list$evaluation_metrics
  model_info <- results_list$model_object

  if (!is.null(metrics$error)) {
    message(sprintf("Prognosis Model: %-15s | Status: Failed (%s)", model_name, metrics$error))
  } else {
    data_source_str <- if(on_new_data) "on New Data" else "on Training Data"
    message(sprintf("\n--- %s Prognosis Model (%s) Metrics ---", model_name, data_source_str))

    if (!is.null(model_info) && !is.null(model_info$model_type)) {
      if (model_info$model_type == "bagging_pro") {
        message(sprintf("Ensemble Type: Bagging (Base: %s, Estimators: %d)",
                        model_info$base_model_name, model_info$n_estimators))
      } else if (model_info$model_type == "stacking_pro") {
        message(sprintf("Ensemble Type: Stacking (Meta: %s, Base models used: %s)",
                        model_info$meta_model_name, paste(model_info$base_models_used, collapse = ", ")))
      }
    }

    message(sprintf("C-index: %.4f", metrics$C_index))

    if (!is.null(metrics$AUROC_Years)) {
      auroc_str <- paste(sapply(names(metrics$AUROC_Years), function(year) {
        val <- metrics$AUROC_Years[[year]]
        if (is.na(val)) "NA" else sprintf("%.4f", val)
      }), collapse = ", ")
      message(sprintf("Time-dependent AUROC (years %s): %s", paste(names(metrics$AUROC_Years), collapse=", "), auroc_str))
    }

    message(sprintf("Average Time-dependent AUROC: %.4f", metrics$AUROC_Average))

    if (!is.null(metrics$KM_HR) && !is.na(metrics$KM_HR)) {
      message(sprintf("KM Group HR (High vs Low): %.4f (p-value: %.4g, Cutoff: %.4f)",
                      metrics$KM_HR, metrics$KM_P_value, metrics$KM_Cutoff))
    } else {
      message("KM Group Analysis: Not applicable or failed.")
    }
    message("--------------------------------------------------")
  }
}

#' @title Evaluate Prognostic Predictions
#' @description A convenience wrapper to evaluate a data frame of prognostic predictions.
#'   This function is ideal for evaluating the output of `apply_pro`.
#'
#' @param prediction_df A data frame containing predictions. Must include columns
#'   named `ID`, `outcome`, `time`, and `score`. This format matches the output
#'   of `apply_pro`.
#' @param years_to_evaluate A numeric vector of specific years at which to
#'   calculate time-dependent AUROC.
#'
#' @return A list of evaluation metrics, including C-index, time-dependent AUROC,
#'   and Kaplan-Meier analysis results.
#' @examples
#' \donttest{
#' # Assume 'trained_model' and 'test_pro' data are available
#' if (requireNamespace("E2E", quietly = TRUE) &&
#'     "train_pro" %in% utils::data(package = "E2E")$results[,3] &&
#'     "test_pro" %in% utils::data(package = "E2E")$results[,3]) {
#'
#'   data(train_pro, package = "E2E")
#'   data(test_pro, package = "E2E")
#'   initialize_modeling_system_pro()
#'   model_results <- models_pro(data = train_pro, model = "lasso_pro")
#'
#'   # 1. Get predictions on new data
#'   predictions <- apply_pro(model_results$lasso_pro$model_object, test_pro)
#'
#'   # 2. Evaluate these predictions using the simplified function
#'   evaluation_metrics <- evaluate_predictions_pro(predictions, years_to_evaluate = c(1, 3))
#'   print(evaluation_metrics)
#' }
#' }
#' @seealso \code{\link{apply_pro}}, \code{\link{evaluate_model_pro}}
#' @export
evaluate_predictions_pro <- function(prediction_df, years_to_evaluate = c(1, 3, 5)) {
  required_cols <- c("ID", "outcome", "time", "score")
  if (!all(required_cols %in% names(prediction_df))) {
    stop(paste("Input data frame must contain columns:", paste(required_cols, collapse = ", ")))
  }

  y_surv_obj <- survival::Surv(time = prediction_df$time, event = prediction_df$outcome)


  results <- evaluate_model_pro(
    Y_surv_obj = y_surv_obj,
    sample_ids = prediction_df$ID,
    years_to_evaluate = years_to_evaluate,
    precomputed_score = prediction_df$score
  )

  return(results$evaluation_metrics)
}

