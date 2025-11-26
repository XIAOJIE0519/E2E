# prognosis.R

# ==============================================================================
# SECTION 0: Global Declarations & Environment Setup
# ==============================================================================

#' @importFrom utils globalVariables
utils::globalVariables(c("x", "y", "recall", "Actual", "Predicted", "Freq", "Percentage",
                         "time", "AUROC", "feature", "value", "ID", "e",
                         "score_col", "label", "sample", "score", ".",
                         "status", "y_surv_", "y_surv_time", "y_surv_event"))

# Internal package environment for model registry.
# This environment acts as a dynamic registry for prognostic model functions,
# allowing for a modular architecture where new models can be registered at runtime.
.model_registry_env_pro <- new.env()
.model_registry_env_pro$known_models_internal <- list()
.model_registry_env_pro$is_initialized <- FALSE

# ==============================================================================
# SECTION 1: Internal Data Preparation & Utility Functions
# ==============================================================================

#' @title Prepare Data for Prognostic Survival Analysis (Internal)
#' @description Prepares and validates input data for time-to-event (survival) analysis.
#'   It standardizes the input dataframe into a consistent format, handling ID extraction,
#'   outcome/time validation, unit conversion, and feature type enforcement.
#'
#' @param data A data frame containing the raw dataset. The expected structure is:
#'   column 1 = Sample ID, column 2 = Binary Outcome (Status), column 3 = Time to Event,
#'   columns 4+ = Features.
#' @param time_unit A character string specifying the unit of the time column.
#'   Options are "day", "month", or "year". Internal calculations are standardized to days.
#'
#' @return A list containing:
#'   \itemize{
#'     \item \code{X}: A data frame of feature predictors.
#'     \item \code{Y_surv}: A \code{Surv} object suitable for survival analysis.
#'     \item \code{sample_ids}: A vector of sample identifiers.
#'     \item \code{outcome_numeric}: Numeric vector of binary outcomes.
#'     \item \code{time_numeric}: Numeric vector of survival times (in days).
#'   }
#' @noRd
.prepare_data_pro <- function(data, time_unit = c("day", "month", "year")) {
  if (!is.data.frame(data)) stop("Input 'data' must be a data frame.")
  if (ncol(data) < 4) stop("Data must have at least 4 columns: ID, Outcome (Status), Time, Features.")

  time_unit <- match.arg(time_unit)

  sample_ids <- data[[1]]
  y_outcome <- base::as.numeric(data[[2]])
  time_val <- base::as.numeric(data[[3]])

  # Scientific unit standardization: Convert all time units to days for consistency
  if (time_unit == "month") {
    time_val <- time_val * (365.25 / 12)
  } else if (time_unit == "year") {
    time_val <- time_val * 365.25
  }

  # Data Integrity Check: Filter invalid survival times or outcomes
  valid_rows <- !is.na(time_val) & !is.na(y_outcome) & time_val > 0
  if (any(!valid_rows)) {
    warning(sprintf("Quality Control: Excluded %d rows with invalid time (<=0 or NA) or missing outcome.", sum(!valid_rows)))
    data <- data[valid_rows, , drop = FALSE]
    sample_ids <- sample_ids[valid_rows]
    y_outcome <- y_outcome[valid_rows]
    time_val <- time_val[valid_rows]
  }

  if (nrow(data) == 0) stop("No valid data remains after quality control cleaning.")
  if (!all(y_outcome %in% c(0, 1))) stop("Outcome status must be binary (0=Censored, 1=Event).")

  Y_surv <- survival::Surv(time = time_val, event = y_outcome)
  X <- data[, -c(1, 2, 3), drop = FALSE]

  # Feature Type Enforcement: Ensure character columns are treated as factors
  for (col_name in names(X)) {
    if (is.character(X[[col_name]])) X[[col_name]] <- base::as.factor(X[[col_name]])
  }

  list(
    X = as.data.frame(X),
    Y_surv = Y_surv,
    sample_ids = sample_ids,
    outcome_numeric = y_outcome,
    time_numeric = time_val
  )
}

#' @title Feature Alignment Utility (Internal)
#' @description Ensures theoretical consistency between training and prediction datasets.
#'   It aligns the feature set of the new data to match the training data's schema,
#'   imputing missing columns with NA and reordering to preserve matrix integrity.
#'
#' @param object A trained model object containing \code{X_train_cols}.
#' @param newdata A data frame of new observations.
#'
#' @return A data frame aligned with the training feature space.
#' @noRd
.ensure_features <- function(object, newdata) {
  if (is.null(object$X_train_cols)) return(newdata)

  needed_cols <- object$X_train_cols
  missing_cols <- setdiff(needed_cols, names(newdata))

  if (length(missing_cols) > 0) {
    # Methodological note: Missing features in inference are filled with NA to maintain structural integrity,
    # though this may impact prediction accuracy depending on the model's handling of missingness.
    for (col in missing_cols) newdata[[col]] <- NA
  }

  # Strictly enforce column order
  return(newdata[, needed_cols, drop = FALSE])
}

#' @title Min-Max Normalization
#' @description Performs linear transformation of data to the range 0 to 1.
#'   Essential for stacking ensembles to normalize risk scores from heterogeneous base learners.
#'
#' @param x A numeric vector.
#' @param min_val Optional reference minimum value (e.g., from training set).
#' @param max_val Optional reference maximum value (e.g., from training set).
#' @return A numeric vector of normalized values.
#' @export
min_max_normalize <- function(x, min_val = NULL, max_val = NULL) {
  if (is.null(min_val)) min_val <- base::min(x, na.rm = TRUE)
  if (is.null(max_val)) max_val <- base::max(x, na.rm = TRUE)
  if (min_val == max_val) return(rep(0.5, length(x)))
  (x - min_val) / (max_val - min_val)
}

# ==============================================================================
# SECTION 2: S3 Prediction Interface (Polymorphic Design)
# ==============================================================================

#' @title Generic Prediction Interface for Prognostic Models
#' @description A unified S3 generic method to generate prognostic risk scores from
#'   various trained model objects. This decouples the prediction implementation
#'   from the high-level evaluation logic, facilitating extensibility.
#'
#' @param object A trained model object with class \code{pro_model}.
#' @param newdata A data frame containing features for prediction.
#' @param ... Additional arguments passed to specific methods.
#' @return A numeric vector representing the prognostic risk score (higher values typically indicate higher risk).
#' @export
predict_pro <- function(object, newdata, ...) {
  UseMethod("predict_pro")
}

#' @export
predict_pro.default <- function(object, newdata, ...) {
  stop("No 'predict_pro' method defined for this model class.")
}

#' @export
predict_pro.survival_glmnet <- function(object, newdata, ...) {
  newdata <- .ensure_features(object, newdata)
  X_matrix <- stats::model.matrix(~ . - 1, data = newdata)
  # 'link' gives the linear predictor (risk score)
  base::as.numeric(stats::predict(object$finalModel, newx = X_matrix, type = "link"))
}

#' @export
predict_pro.survival_rsf <- function(object, newdata, ...) {
  newdata <- .ensure_features(object, newdata)
  df_pred <- cbind(
    data.frame(
      time = rep(1, nrow(newdata)),    # dummy time required by rfsrc interface
      status = rep(0, nrow(newdata))   # dummy status
    ),
    newdata
  )
  pred_obj <- randomForestSRC::predict.rfsrc(object$finalModel, newdata = df_pred)
  raw_score <- pred_obj$predicted

  # If the model was identified as having inverted concordance during training, invert the score
  if (isTRUE(object$inverted)) {
    return(-raw_score)
  } else {
    return(raw_score)
  }
}

#' @export
predict_pro.survival_stepcox <- function(object, newdata, ...) {
  newdata <- .ensure_features(object, newdata)
  stats::predict(object$finalModel, newdata = newdata, type = "lp")
}

#' @export
predict_pro.survival_gbm <- function(object, newdata, ...) {
  newdata <- .ensure_features(object, newdata)
  stats::predict(object$finalModel, newdata = newdata,
                 n.trees = object$finalModel$best_iter, type = "link")
}

#' @export
predict_pro.survival_coxboost <- function(object, newdata, ...) {
  newdata <- .ensure_features(object, newdata)
  X_matrix <- as.matrix(newdata)
  as.vector(predict(object$finalModel, newdata = X_matrix, type = "lp"))
}

#' @export
predict_pro.survival_plsRcox <- function(object, newdata, ...) {
  newdata <- .ensure_features(object, newdata)
  X_matrix <- as.matrix(newdata)
  stats::predict(object$finalModel, newdata = X_matrix, type = "lp")
}

#' @export
predict_pro.bagging_pro <- function(object, newdata, ...) {
  base_models <- object$base_model_objects
  n_samples <- nrow(newdata)
  n_estimators <- length(base_models)

  all_scores <- matrix(NA, nrow = n_samples, ncol = n_estimators)

  for (i in seq_len(n_estimators)) {
    if (!is.null(base_models[[i]])) {
      all_scores[, i] <- predict_pro(base_models[[i]], newdata)
    }
  }
  # Consensus via averaging risk scores
  rowMeans(all_scores, na.rm = TRUE)
}

#' @export
predict_pro.stacking_pro <- function(object, newdata, ...) {
  base_models <- object$base_model_objects
  n_samples <- nrow(newdata)

  # 1. Generate Base Model Predictions
  all_base_scores <- matrix(NA, nrow = n_samples, ncol = length(base_models))
  colnames(all_base_scores) <- object$base_models_used

  for (i in seq_along(base_models)) {
    if (!is.null(base_models[[i]])) {
      all_base_scores[, i] <- predict_pro(base_models[[i]], newdata)
    }
  }

  # 2. Normalize Base Scores (Using Training Parameters for consistency)
  meta_params <- object$meta_normalize_params
  if (!is.null(meta_params)) {
    for (name in colnames(all_base_scores)) {
      if (name %in% names(meta_params)) {
        params <- meta_params[[name]]
        all_base_scores[, name] <- min_max_normalize(
          all_base_scores[, name], params$min_val, params$max_val
        )
      }
    }
  }

  # 3. Create Meta Features DataFrame
  X_meta_new <- as.data.frame(all_base_scores)
  names(X_meta_new) <- paste0("pred_", object$base_models_used)

  # 4. Meta Model Prediction
  predict_pro(object$trained_meta_model, X_meta_new)
}

# ==============================================================================
# SECTION 3: Training Functions (Core Algorithms)
# ==============================================================================

#' @title Train Lasso Cox Proportional Hazards Model
#' @description Fits a Cox proportional hazards model regularized by the Lasso (L1) penalty.
#'   Uses cross-validation to select the optimal lambda.
#'
#' @param X A data frame of predictors.
#' @param y_surv A \code{Surv} object containing time and status.
#' @param tune Logical. If TRUE, performs internal tuning (currently handled by cv.glmnet automatically).
#'
#' @return An object of class \code{survival_glmnet} and \code{pro_model}.
#' @examples
#' \donttest{
#'   library(survival)
#'   # Create dummy data
#'   set.seed(123)
#'   df <- data.frame(time = rexp(50), status = sample(0:1, 50, replace=TRUE),
#'                    var1 = rnorm(50), var2 = rnorm(50))
#'   y <- Surv(df$time, df$status)
#'   x <- df[, c("var1", "var2")]
#'
#'   model <- lasso_pro(x, y)
#'   print(class(model))
#' }
#' @importFrom glmnet cv.glmnet glmnet
#' @importFrom stats model.matrix predict
#' @export
lasso_pro <- function(X, y_surv, tune = FALSE) {
  X_matrix <- stats::model.matrix(~ . - 1, data = X)
  cv_fit <- glmnet::cv.glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 1)
  final_model <- glmnet::glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 1, lambda = cv_fit$lambda.min)
  final_model$fitted_scores <- base::as.numeric(stats::predict(final_model, newx = X_matrix, type = "link"))

  structure(
    list(finalModel = final_model, X_train_cols = colnames(X), model_type = "survival_glmnet"),
    class = c("survival_glmnet", "pro_model")
  )
}

#' @title Train Elastic Net Cox Model
#' @description Fits a Cox model with Elastic Net regularization (mixture of L1 and L2 penalties).
#'   Alpha is fixed at 0.5.
#'
#' @inheritParams lasso_pro
#' @return An object of class \code{survival_glmnet} and \code{pro_model}.
#' @export
en_pro <- function(X, y_surv, tune = FALSE) {
  X_matrix <- stats::model.matrix(~ . - 1, data = X)
  cv_fit <- glmnet::cv.glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 0.5)
  final_model <- glmnet::glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 0.5, lambda = cv_fit$lambda.min)
  final_model$fitted_scores <- base::as.numeric(stats::predict(final_model, newx = X_matrix, type = "link"))

  structure(
    list(finalModel = final_model, X_train_cols = colnames(X), model_type = "survival_glmnet"),
    class = c("survival_glmnet", "pro_model")
  )
}

#' @title Train Ridge Cox Model
#' @description Fits a Cox model with Ridge (L2) regularization.
#'
#' @inheritParams lasso_pro
#' @return An object of class \code{survival_glmnet} and \code{pro_model}.
#' @export
ridge_pro <- function(X, y_surv, tune = FALSE) {
  X_matrix <- stats::model.matrix(~ . - 1, data = X)
  cv_fit <- glmnet::cv.glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 0)
  final_model <- glmnet::glmnet(x = X_matrix, y = y_surv, family = "cox", alpha = 0, lambda = cv_fit$lambda.min)
  final_model$fitted_scores <- base::as.numeric(stats::predict(final_model, newx = X_matrix, type = "link"))

  structure(
    list(finalModel = final_model, X_train_cols = colnames(X), model_type = "survival_glmnet"),
    class = c("survival_glmnet", "pro_model")
  )
}

#' @title Train Random Survival Forest (RSF)
#' @description Fits a Random Survival Forest using the log-rank splitting rule.
#'   Includes capabilities for hyperparameter tuning via grid search over \code{ntree},
#'   \code{nodesize}, and \code{mtry}.
#'
#' @param X A data frame of predictors.
#' @param y_surv A \code{Surv} object containing time and status.
#' @param tune Logical. If TRUE, performs grid search for optimal hyperparameters based on C-index.
#' @param tune_params Optional data frame containing the grid for tuning.
#'
#' @return An object of class \code{survival_rsf} and \code{pro_model}.
#' @importFrom randomForestSRC rfsrc predict.rfsrc
#' @importFrom survcomp concordance.index
#' @export
rsf_pro <- function(X, y_surv, tune = FALSE, tune_params = NULL) {
  if (nrow(X) < 5 || sum(y_surv[,2]) < 2) return(NULL)

  time_vals <- as.numeric(y_surv[, 1])
  status_vals <- as.numeric(y_surv[, 2])

  data_for_rsf <- cbind(
    data.frame(time = time_vals, status = status_vals),
    X
  )

  formula_str <- "Surv(time, status) ~ ."

  # --- Hyperparameter Tuning Logic ---
  if (tune) {
    if (is.null(tune_params)) {
      tune_grid <- expand.grid(
        ntree = c(500, 1000, 1500),
        nodesize = c(5, 10, 15, 20),
        mtry = c(max(1, floor(ncol(X)/3)),
                 max(1, floor(sqrt(ncol(X)))),
                 max(1, floor(ncol(X)/2)))
      )
    } else {
      tune_grid <- tune_params
    }

    message(sprintf("RSF Tuning: Testing %d parameter combinations", nrow(tune_grid)))

    best_cindex <- -Inf
    best_params <- NULL

    for (i in 1:nrow(tune_grid)) {
      params <- tune_grid[i, ]
      fit_tmp <- tryCatch({
        randomForestSRC::rfsrc(
          formula = stats::formula(formula_str),
          data = data_for_rsf,
          ntree = params$ntree,
          nodesize = params$nodesize,
          mtry = params$mtry,
          splitrule = "logrank",
          importance = FALSE,
          proximity = FALSE,
          forest = TRUE
        )
      }, error = function(e) NULL)

      if (!is.null(fit_tmp) && !is.null(fit_tmp$predicted.oob)) {
        cindex_tmp <- tryCatch({
          survcomp::concordance.index(
            x = fit_tmp$predicted.oob,
            surv.time = time_vals,
            surv.event = status_vals
          )$c.index
        }, error = function(e) 0)

        # Directionality check: RSF usually outputs mortality, but check for concordance inversion
        if (cindex_tmp < 0.5) cindex_tmp <- 1 - cindex_tmp

        if (cindex_tmp > best_cindex) {
          best_cindex <- cindex_tmp
          best_params <- params
        }
      }
    }

    if (is.null(best_params)) {
      warning("RSF Tuning failed, using default parameters.")
      best_params <- list(ntree = 1000, nodesize = 15, mtry = max(1, floor(sqrt(ncol(X)))))
    } else {
      message(sprintf("Best RSF params: ntree=%d, nodesize=%d, mtry=%d (C-index=%.3f)",
                      best_params$ntree, best_params$nodesize, best_params$mtry, best_cindex))
    }

  } else {
    best_params <- list(ntree = 1000, nodesize = 15, mtry = max(1, floor(sqrt(ncol(X)))))
  }

  fit <- tryCatch({
    randomForestSRC::rfsrc(
      formula = stats::formula(formula_str),
      data = data_for_rsf,
      ntree = best_params$ntree,
      nodesize = best_params$nodesize,
      mtry = best_params$mtry,
      splitrule = "logrank",
      importance = TRUE,
      proximity = TRUE,
      forest = TRUE
    )
  }, error = function(e) {
    stop(paste("RSF training failed:", e$message))
  })

  # --- Directionality Verification ---
  # Ensure higher scores correlate with higher risk (Event).
  inverted <- FALSE
  if (!is.null(fit$predicted.oob) && !all(is.na(fit$predicted.oob))) {
    if (stats::sd(fit$predicted.oob, na.rm = TRUE) > 1e-6) {
      c_index_val <- tryCatch({
        survcomp::concordance.index(
          x = fit$predicted.oob,
          surv.time = time_vals,
          surv.event = status_vals
        )$c.index
      }, error = function(e) 0.5)

      if (!is.na(c_index_val) && c_index_val < 0.5) {
        inverted <- TRUE
      }
    }
  }

  raw_score <- if(is.null(fit$predicted.oob)) rep(NA, nrow(X)) else fit$predicted.oob
  fit$fitted_scores <- if(inverted) -raw_score else raw_score

  structure(
    list(
      finalModel = fit,
      X_train_cols = colnames(X),
      model_type = "survival_rsf",
      inverted = inverted,
      best_hyperparams = best_params
    ),
    class = c("survival_rsf", "pro_model")
  )
}

#' @title Train Stepwise Cox Model (AIC-based)
#' @description Fits a Cox model and performs backward stepwise selection based on AIC.
#'
#' @inheritParams lasso_pro
#' @return An object of class \code{survival_stepcox} and \code{pro_model}.
#' @importFrom MASS stepAIC
#' @importFrom survival coxph
#' @export
stepcox_pro <- function(X, y_surv, tune = FALSE) {
  data_for_cox <- cbind(y_surv_ = y_surv, X)
  fit_full <- survival::coxph(stats::as.formula(paste("y_surv_ ~", paste(colnames(X), collapse = " + "))), data = data_for_cox)
  fit <- MASS::stepAIC(fit_full, direction = "backward", trace = FALSE)
  fit$fitted_scores <- stats::predict(fit, newdata = X, type = "lp")

  structure(
    list(finalModel = fit, X_train_cols = colnames(X), model_type = "survival_stepcox"),
    class = c("survival_stepcox", "pro_model")
  )
}

#' @title Train Gradient Boosting Machine (GBM) for Survival
#' @description Fits a stochastic gradient boosting model using the Cox Partial Likelihood distribution.
#'   Supports random search for hyperparameter optimization.
#'
#' @param X A data frame of predictors.
#' @param y_surv A \code{Surv} object.
#' @param tune Logical. If TRUE, performs random search.
#' @param cv.folds Integer. Number of cross-validation folds.
#' @param max_tune_iter Integer. Maximum iterations for random search.
#'
#' @return An object of class \code{survival_gbm} and \code{pro_model}.
#' @importFrom gbm gbm gbm.perf
#' @export
gbm_pro <- function(X, y_surv, tune = FALSE, cv.folds = 5, max_tune_iter = 10) {
  data_for_gbm <- cbind(y_surv_time = y_surv[,1], y_surv_event = y_surv[,2], X)

  if (tune) {
    sample_params <- function() {
      list(
        n.trees = sample(c(100, 200, 300, 500, 800), 1),
        interaction.depth = sample(c(2, 3, 4, 5), 1),
        shrinkage = sample(c(0.001, 0.005, 0.01, 0.05, 0.1), 1),
        n.minobsinnode = sample(c(5, 10, 15, 20), 1)
      )
    }

    message(sprintf("GBM Random Search: Testing up to %d configurations", max_tune_iter))

    best_cindex <- -Inf
    best_params <- NULL
    best_iter <- NULL
    tested_configs <- 0

    for (i in 1:max_tune_iter) {
      params <- sample_params()
      tested_configs <- tested_configs + 1

      fit_tmp <- tryCatch({
        gbm::gbm(
          formula = survival::Surv(y_surv_time, y_surv_event) ~ .,
          data = data_for_gbm,
          distribution = "coxph",
          n.trees = params$n.trees,
          interaction.depth = params$interaction.depth,
          shrinkage = params$shrinkage,
          n.minobsinnode = params$n.minobsinnode,
          cv.folds = cv.folds,
          verbose = FALSE
        )
      }, error = function(e) NULL)

      if (!is.null(fit_tmp)) {
        best_iter_tmp <- gbm::gbm.perf(fit_tmp, method = "cv", plot.it = FALSE)
        pred_score <- predict(fit_tmp, newdata = X, n.trees = best_iter_tmp, type = "link")

        cindex_tmp <- tryCatch({
          survcomp::concordance.index(
            x = pred_score,
            surv.time = y_surv[,1],
            surv.event = y_surv[,2]
          )$c.index
        }, error = function(e) 0)

        if (cindex_tmp > best_cindex) {
          best_cindex <- cindex_tmp
          best_params <- params
          best_iter <- best_iter_tmp
        }
      }
    }

    if (is.null(best_params)) {
      warning("GBM tuning failed, using defaults.")
      best_params <- list(n.trees = 100, interaction.depth = 3,
                          shrinkage = 0.1, n.minobsinnode = 10)
    } else {
      message(sprintf("Best GBM (iter %d/%d): trees=%d, depth=%d, lr=%.3f, best_iter=%d (C=%.3f)",
                      tested_configs, max_tune_iter,
                      best_params$n.trees, best_params$interaction.depth,
                      best_params$shrinkage, best_iter, best_cindex))
    }

  } else {
    best_params <- list(
      n.trees = 100,
      interaction.depth = 3,
      shrinkage = 0.1,
      n.minobsinnode = 10
    )
    best_iter <- NULL
  }

  fit <- gbm::gbm(
    formula = survival::Surv(y_surv_time, y_surv_event) ~ .,
    data = data_for_gbm,
    distribution = "coxph",
    n.trees = best_params$n.trees,
    interaction.depth = best_params$interaction.depth,
    shrinkage = best_params$shrinkage,
    n.minobsinnode = best_params$n.minobsinnode,
    cv.folds = cv.folds
  )

  final_best_iter <- if(!is.null(best_iter)) {
    best_iter
  } else {
    gbm::gbm.perf(fit, method = "cv", plot.it = FALSE)
  }

  fit$best_iter <- final_best_iter
  fit$fitted_scores <- predict(fit, newdata = X, n.trees = final_best_iter, type = "link")

  structure(
    list(
      finalModel = fit,
      X_train_cols = colnames(X),
      model_type = "survival_gbm",
      best_hyperparams = c(best_params, list(best_iter = final_best_iter))
    ),
    class = c("survival_gbm", "pro_model")
  )
}

#' @title Train CoxBoost
#' @description Fits a Cox model by likelihood based boosting.
#'
#' @inheritParams lasso_pro
#' @return An object of class \code{survival_coxboost} and \code{pro_model}.
#' @export
cb_pro <- function(X, y_surv, tune = FALSE) {
  X_matrix <- as.matrix(X)
  stepno_val <- 100
  if (tune) stepno_val <- 100 # Placeholder for advanced tuning implementation

  # Note: CoxBoost requires specific namespace loading in some R versions
  fit <- CoxBoost::CoxBoost(time = y_surv[,1], status = y_surv[,2], x = X_matrix, stepno = stepno_val, penalty = 100)
  fit$fitted_scores <- as.vector(predict(fit, newdata = X_matrix, type = "lp"))

  structure(
    list(finalModel = fit, X_train_cols = colnames(X), model_type = "survival_coxboost"),
    class = c("survival_coxboost", "pro_model")
  )
}

#' @title Train Partial Least Squares Cox (PLS-Cox)
#' @description Fits a Cox model using Partial Least Squares reduction for high-dimensional data.
#'
#' @inheritParams lasso_pro
#' @return An object of class \code{survival_plsRcox} and \code{pro_model}.
#' @importFrom plsRcox plsRcox
#' @export
pls_pro <- function(X, y_surv, tune = FALSE) {
  X_matrix <- as.matrix(X)
  # Basic heuristic for number of components
  nt_val <- min(3, ncol(X_matrix) - 1)
  fit <- plsRcox::plsRcox(Xplan = X_matrix, time = y_surv[,1], event = y_surv[,2], nt = nt_val)
  fit$fitted_scores <- stats::predict(fit, newdata = X_matrix, type = "lp")

  structure(
    list(finalModel = fit, X_train_cols = colnames(X), model_type = "survival_plsRcox"),
    class = c("survival_plsRcox", "pro_model")
  )
}

# ==============================================================================
# SECTION 4: Ensemble Implementations (Bagging & Stacking)
# ==============================================================================

#' @title Train Bagging Ensemble for Prognosis
#' @description Implements Bootstrap Aggregating (Bagging) for survival models.
#'   It trains multiple base models on bootstrapped subsets and averages the risk scores.
#'   This method reduces variance and improves stability.
#'
#' @param data Input data frame (ID, Status, Time, Features).
#' @param base_model_name Character string name of the base model (e.g., "rsf_pro").
#' @param n_estimators Integer. Number of bootstrap iterations.
#' @param subset_fraction Numeric (0-1). Fraction of data to sample in each iteration.
#' @param tune_base_model Logical. Whether to tune each base model (computationally expensive).
#' @param time_unit Time unit of the input data.
#' @param years_to_evaluate Numeric vector of years for time-dependent AUC evaluation.
#' @param seed Integer seed for reproducibility.
#'
#' @return A list containing the ensemble object, sample scores, and evaluation metrics.
#' @export
bagging_pro <- function(data, base_model_name, n_estimators = 10, subset_fraction = 0.632,
                        tune_base_model = FALSE, time_unit = "day", years_to_evaluate = c(1, 3, 5), seed = 456) {

  if (!.model_registry_env_pro$is_initialized) initialize_modeling_system_pro()
  set.seed(seed)
  data_prepared <- .prepare_data_pro(data, time_unit)
  X_data <- data_prepared$X
  Y_surv_obj <- data_prepared$Y_surv

  base_model_func <- get_registered_models_pro()[[base_model_name]]
  valid_models <- list()

  max_attempts <- n_estimators * 5
  attempts <- 0
  success_count <- 0
  last_error_msg <- "Unknown error"

  message(sprintf("Running Bagging: Target %d models using %s...", n_estimators, base_model_name))

  while(success_count < n_estimators && attempts < max_attempts) {
    attempts <- attempts + 1
    current_seed <- seed + attempts
    set.seed(current_seed)

    indices <- sample(1:nrow(X_data), floor(nrow(X_data) * subset_fraction), replace = TRUE)
    X_subset <- X_data[indices, , drop=FALSE]
    Y_subset <- Y_surv_obj[indices]

    # Validity Check: Ensure sufficient events in bootstrap sample
    n_events <- sum(Y_subset[,2])
    n_samples <- nrow(X_subset)

    if (n_samples < 10 || n_events < 3) {
      next
    }

    # Variance Check: Ensure features are not constant
    valid_cols <- sapply(X_subset, function(x) length(unique(x)) > 1)
    if (sum(valid_cols) < 2) {
      next
    }

    model <- tryCatch({
      suppressWarnings({
        base_model_func(X_subset, Y_subset, tune = tune_base_model)
      })
    }, error = function(e) {
      last_error_msg <<- e$message
      return(NULL)
    })

    if (!is.null(model) && inherits(model, "pro_model")) {
      success_count <- success_count + 1
      valid_models[[success_count]] <- model
    }
  }

  if (length(valid_models) == 0) {
    stop(sprintf("Bagging failed for %s. Last captured error: %s. (Attempted %d times)",
                 base_model_name, last_error_msg, max_attempts))
  }

  bagging_obj <- structure(
    list(
      base_model_name = base_model_name,
      n_estimators = length(valid_models),
      base_model_objects = valid_models,
      X_train_cols = colnames(X_data),
      model_type = "bagging_pro"
    ),
    class = c("bagging_pro", "pro_model")
  )

  eval_results <- evaluate_model_pro(bagging_obj, X_data, Y_surv_obj, data_prepared$sample_ids, years_to_evaluate)

  list(model_object = bagging_obj, sample_score = eval_results$sample_score, evaluation_metrics = eval_results$evaluation_metrics)
}

#' @title Train Stacking Ensemble for Prognosis
#' @description Implements a Stacking Ensemble (Super Learner).
#'   It uses the risk scores from top-performing base models as meta-features
#'   to train a second-level meta-learner.
#'
#' @param results_all_models List of results from \code{models_pro()}.
#' @param data Training data.
#' @param meta_model_name Name of the meta-learner (e.g., "lasso_pro").
#' @param top Integer. Number of top base models to include based on C-index.
#' @param tune_meta Logical. Tune the meta-learner?
#' @param time_unit Time unit.
#' @param years_to_evaluate Evaluation years.
#' @param seed Integer seed.
#'
#' @return A list containing the stacking object and evaluation results.
#' @export
stacking_pro <- function(results_all_models, data, meta_model_name, top = 3,
                         tune_meta = FALSE, time_unit = "day", years_to_evaluate = c(1, 3, 5), seed = 789) {

  if (!.model_registry_env_pro$is_initialized) initialize_modeling_system_pro()

  set.seed(seed)
  data_prepared <- .prepare_data_pro(data, time_unit)
  Y_surv_obj <- data_prepared$Y_surv

  # 1. Selection Strategy: Choose top models by C-index
  c_indices <- sapply(results_all_models, function(x) x$evaluation_metrics$C_index)
  top_models <- names(sort(c_indices, decreasing = TRUE))[1:min(top, length(c_indices))]
  selected_objects <- lapply(results_all_models[top_models], `[[`, "model_object")

  # 2. Meta-Feature Generation & Normalization
  # We use min-max normalization to make scores comparable before feeding to meta-learner
  X_meta <- data.frame(ID = data_prepared$sample_ids)
  meta_normalize_params <- list()

  for (name in top_models) {
    scores <- results_all_models[[name]]$sample_score$score
    min_v <- min(scores, na.rm = TRUE)
    max_v <- max(scores, na.rm = TRUE)
    meta_normalize_params[[name]] <- list(min_val = min_v, max_val = max_v)

    col_name <- paste0("pred_", name)
    X_meta[[col_name]] <- min_max_normalize(scores, min_v, max_v)
  }
  X_meta_features <- X_meta[, -1, drop = FALSE] # Remove ID

  # 3. Meta-Model Training
  meta_func <- get_registered_models_pro()[[meta_model_name]]
  meta_mdl <- meta_func(X_meta_features, Y_surv_obj, tune = tune_meta)

  stacking_obj <- structure(
    list(
      meta_model_name = meta_model_name,
      base_models_used = top_models,
      base_model_objects = selected_objects,
      trained_meta_model = meta_mdl,
      meta_normalize_params = meta_normalize_params,
      X_train_cols = colnames(data_prepared$X),
      model_type = "stacking_pro"
    ),
    class = c("stacking_pro", "pro_model")
  )

  eval_results <- evaluate_model_pro(stacking_obj, data_prepared$X, Y_surv_obj, data_prepared$sample_ids, years_to_evaluate)

  list(model_object = stacking_obj, sample_score = eval_results$sample_score, evaluation_metrics = eval_results$evaluation_metrics)
}

# ==============================================================================
# SECTION 5: Evaluation Framework
# ==============================================================================

#' @title Evaluate Prognostic Model Performance
#' @description Comprehensive evaluation of survival models using:
#'   1. Harrell's Concordance Index (C-index).
#'   2. Time-dependent Area Under the ROC Curve (AUROC) at specified years.
#'   3. Kaplan-Meier analysis comparing high vs. low risk groups (based on median split).
#'
#' @param trained_model_obj A trained model object (optional if precomputed_score provided).
#' @param X_data Features for prediction (optional if precomputed_score provided).
#' @param Y_surv_obj True survival object.
#' @param sample_ids Vector of IDs.
#' @param years_to_evaluate Numeric vector of years for time-dependent AUC.
#' @param precomputed_score Numeric vector of pre-calculated risk scores.
#' @param meta_normalize_params Internal use.
#'
#' @return A list containing a dataframe of scores and a list of evaluation metrics.
#' @importFrom survcomp concordance.index
#' @importFrom survivalROC survivalROC
#' @importFrom survival coxph
#' @export
evaluate_model_pro <- function(trained_model_obj = NULL, X_data = NULL, Y_surv_obj, sample_ids,
                               years_to_evaluate = c(1, 3, 5),
                               precomputed_score = NULL,
                               meta_normalize_params = NULL) {

  score <- precomputed_score

  # 1. Inference: Obtain Scores via Prediction Interface if not provided
  if (is.null(score)) {
    if (is.null(trained_model_obj) || is.null(X_data)) {
      stop("Either 'precomputed_score' or ('trained_model_obj' + 'X_data') is required.")
    }

    score <- tryCatch({
      predict_pro(trained_model_obj, newdata = X_data)
    }, error = function(e) {
      stop(sprintf("Prediction failed for model class '%s': %s", class(trained_model_obj)[1], e$message))
    })
  }

  score[is.na(score)] <- stats::median(score, na.rm = TRUE)

  # 2. Metrics: C-Index
  c_index_val <- tryCatch({
    survcomp::concordance.index(x = score, surv.time = Y_surv_obj[,1], surv.event = Y_surv_obj[,2])$c.index
  }, error = function(e) { NA })

  # 3. Metrics: Time-dependent AUROC
  auroc_yearly <- list()
  for (year in years_to_evaluate) {
    eval_time <- year * 365.25
    if (max(Y_surv_obj[,1], na.rm=TRUE) < eval_time) {
      auroc_yearly[[as.character(year)]] <- NA
    } else {
      roc <- tryCatch({
        survivalROC::survivalROC(Stime = Y_surv_obj[,1], status = Y_surv_obj[,2], marker = score,
                                 predict.time = eval_time, method = "NNE", span = 0.25)$AUC
      }, error = function(e) NA)
      auroc_yearly[[as.character(year)]] <- roc
    }
  }

  # 4. Metrics: Kaplan-Meier Risk Stratification
  median_score <- stats::median(score, na.rm = TRUE)
  risk_group <- factor(ifelse(score > median_score, "High", "Low"), levels = c("Low", "High"))

  km_stats <- list(hr = NA, p = NA, cutoff = NA)
  if (length(unique(risk_group)) == 2) {
    try({
      cox <- survival::coxph(Y_surv_obj ~ risk_group)
      km_stats$hr <- summary(cox)$conf.int[1, "exp(coef)"]
      km_stats$p <- summary(cox)$coefficients[1, "Pr(>|z|)"]
      km_stats$cutoff <- median_score
    }, silent = TRUE)
  }

  list(
    sample_score = data.frame(ID = sample_ids, outcome = Y_surv_obj[,2], time = Y_surv_obj[,1], score = score),
    evaluation_metrics = list(C_index = c_index_val, AUROC_Years = auroc_yearly,
                              AUROC_Average = mean(unlist(auroc_yearly), na.rm = TRUE),
                              KM_HR = km_stats$hr, KM_P_value = km_stats$p, KM_Cutoff = km_stats$cutoff)
  )
}

# ==============================================================================
# SECTION 6: High-Level APIs & System Management
# ==============================================================================

#' @title Register a Prognostic Model
#' @description Registers a model function into the internal system environment, making it available for batch execution.
#' @param name String identifier for the model.
#' @param func The model training function.
#' @export
register_model_pro <- function(name, func) {
  .model_registry_env_pro$known_models_internal[[name]] <- func
}

#' @title Get Registered Prognostic Models
#' @description Retrieves the list of available models.
#' @return Named list of functions.
#' @export
get_registered_models_pro <- function() {
  .model_registry_env_pro$known_models_internal
}

#' @title Initialize Prognosis Modeling System
#' @description Initializes the environment and registers default survival models
#'   (Lasso, Elastic Net, Ridge, RSF, StepCox, GBM, CoxBoost, PLS).
#' @export
initialize_modeling_system_pro <- function() {
  if (.model_registry_env_pro$is_initialized) return(invisible(NULL))

  register_model_pro("lasso_pro", lasso_pro)
  register_model_pro("en_pro", en_pro)
  register_model_pro("ridge_pro", ridge_pro)
  register_model_pro("rsf_pro", rsf_pro)
  register_model_pro("stepcox_pro", stepcox_pro)
  register_model_pro("gbm_pro", gbm_pro)
  register_model_pro("cb_pro", cb_pro)
  register_model_pro("pls_pro", pls_pro)

  .model_registry_env_pro$is_initialized <- TRUE
  message("Prognosis modeling system initialized.")
}

#' @title Run Multiple Prognostic Models
#' @description High-level API to train and evaluate multiple survival models in batch.
#'
#' @param data Input data frame.
#' @param model Character vector of model names or "all_pro".
#' @param tune Logical. Enable hyperparameter tuning?
#' @param seed Random seed.
#' @param time_unit Time unit of input.
#' @param years_to_evaluate Years for AUC calculation.
#'
#' @return A list of model results.
#' @export
models_pro <- function(data, model = "all_pro", tune = FALSE, seed = 123, time_unit = "day", years_to_evaluate = c(1, 3, 5)) {
  if (!.model_registry_env_pro$is_initialized) stop("System not initialized. Run initialize_modeling_system_pro() first.")

  all_models <- get_registered_models_pro()
  models_to_run <- if(length(model)==1 && model=="all_pro") names(all_models) else model

  set.seed(seed)
  prep <- .prepare_data_pro(data, time_unit)
  results <- list()

  for (m in models_to_run) {
    message(sprintf("Running model: %s", m))
    set.seed(seed)
    # Training
    mdl <- tryCatch(all_models[[m]](prep$X, prep$Y_surv, tune = tune), error = function(e) NULL)

    if (!is.null(mdl)) {
      # Evaluation
      eval_res <- evaluate_model_pro(mdl, prep$X, prep$Y_surv, prep$sample_ids, years_to_evaluate)
      results[[m]] <- list(model_object = mdl, sample_score = eval_res$sample_score, evaluation_metrics = eval_res$evaluation_metrics)
    } else {
      results[[m]] <- list(error = "Training Failed")
    }
  }
  results
}

#' @title Apply Prognostic Model to New Data
#' @description Generates risk scores for new patients using a trained model.
#'
#' @param trained_model_object A trained object (class \code{pro_model}).
#' @param new_data Data frame of new patients.
#' @param time_unit Time unit for data preparation.
#'
#' @return Data frame with IDs, outcomes (if available), and risk scores.
#' @export
apply_pro <- function(trained_model_object, new_data, time_unit = "day") {
  message("Applying model on new data...")
  # Note: .prepare_data_pro expects outcome columns. If purely inference data (no outcome),
  # ensure dummy columns are added before calling, or extend .prepare_data to handle inference mode.
  # Here we assume validation data structure matches training.
  prep <- .prepare_data_pro(new_data, time_unit)

  scores <- predict_pro(trained_model_object, prep$X)

  data.frame(
    ID = prep$sample_ids,
    outcome = prep$outcome_numeric,
    time = prep$time_numeric,
    score = scores
  )
}

#' @title Evaluate External Predictions
#' @description Calculates performance metrics for external prediction sets.
#'
#' @param prediction_df Data frame with columns \code{time}, \code{outcome}, \code{score}, \code{ID}.
#' @param years_to_evaluate Years for AUC.
#'
#' @return List of evaluation metrics.
#' @export
evaluate_predictions_pro <- function(prediction_df, years_to_evaluate = c(1, 3, 5)) {
  y_surv <- survival::Surv(prediction_df$time, prediction_df$outcome)
  res <- evaluate_model_pro(Y_surv_obj = y_surv, sample_ids = prediction_df$ID,
                            years_to_evaluate = years_to_evaluate, precomputed_score = prediction_df$score)
  res$evaluation_metrics
}

#' @title Print Prognostic Model Summary
#' @description Formatted console output of model performance.
#' @param model_name Name of the model.
#' @param results_list Result object containing \code{evaluation_metrics}.
#' @export
print_model_summary_pro <- function(model_name, results_list) {
  metrics <- results_list$evaluation_metrics
  if (is.null(metrics)) {
    message(paste("Model", model_name, "failed."))
    return()
  }
  message(sprintf("\n--- %s Summary ---", model_name))
  message(sprintf("C-index: %.4f", metrics$C_index))
  message(sprintf("Avg AUC: %.4f", metrics$AUROC_Average))
  message(sprintf("KM HR: %.4f (p=%.4g)", metrics$KM_HR, metrics$KM_P_value))
}
