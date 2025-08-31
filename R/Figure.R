# Figure.R

# Global Aesthetic Color Settings
# These colors are used consistently across plotting functions for branding and clarity.
primary_color <- "#2E86AB"   # Deep blue, often used for primary lines or fills.
secondary_color <- "#A23B72" # Magenta, used for secondary elements or contrasts.
accent_color <- "#F18F01"    # Orange, used for highlighting specific points or annotations.

#' @importFrom utils globalVariables
utils::globalVariables(c("FPR", "TPR", "TimePoint", "Predicted", "Actual", "Freq", "Percentage", "precision", "Label"))

# ------------------------------------------------------------------------------
# 1. Diagnostic Model Visualization Function (figure_dia)
# ------------------------------------------------------------------------------

#' @title Plot Diagnostic Model Evaluation Figures
#' @description Generates and returns a ggplot object for Receiver Operating
#'   Characteristic (ROC) curves, Precision-Recall (PRC) curves, or confusion matrices.
#'
#' @param type String, specifies the type of plot to generate. Options are
#'   "roc", "prc", or "matrix".
#' @param data A list object containing model evaluation results. It must include:
#'   \itemize{
#'     \item `sample_score`: A data frame with "label" (0/1) and "score" columns.
#'     \item `evaluation_metrics`: A list with a "Final_Threshold" or "Final_Threshold" value.
#'   }
#' @param file Optional. A string specifying the path to save the plot (e.g.,
#'   "plot.png"). If `NULL` (the default), the plot object is returned instead of being saved.
#'
#' @return A ggplot object. If the `file` argument is provided, the plot is also
#'   saved to the specified path.
#' @examples
#' # Create example data for a diagnostic model
#' external_eval_example_dia <- list(
#'   sample_score = data.frame(
#'     ID = paste0("S", 1:100),
#'     label = sample(c(0, 1), 100, replace = TRUE),
#'     score = runif(100, 0, 1)
#'   ),
#'   evaluation_metrics = list(
#'     Final_Threshold = 0.53
#'   )
#' )
#'
#' # Generate an ROC curve plot object
#' roc_plot <- figure_dia(type = "roc", data = external_eval_example_dia)
#' # To display the plot, simply run:
#' # print(roc_plot)
#'
#' # Generate a PRC curve and save it to a temporary file
#' # tempfile() creates a safe, temporary path as required by CRAN
#' temp_prc_path <- tempfile(fileext = ".png")
#' figure_dia(type = "prc", data = external_eval_example_dia, file = temp_prc_path)
#'
#' # Generate a Confusion Matrix plot
#' matrix_plot <- figure_dia(type = "matrix", data = external_eval_example_dia)
#'
#' @importFrom pROC roc coords
#' @importFrom PRROC pr.curve
#' @importFrom ggplot2 ggplot aes geom_line geom_abline geom_point annotate labs
#'   scale_x_continuous scale_y_continuous theme_bw element_text element_blank
#'   geom_tile geom_text scale_fill_gradient scale_x_discrete scale_y_discrete
#'   theme_minimal coord_fixed ggsave theme
#' @importFrom dplyr select
#' @export
figure_dia <- function(type, data, file = NULL) {

  if (!type %in% c("roc", "prc", "matrix")) {
    stop("Invalid 'type'. Choose from 'roc', 'prc', or 'matrix'.")
  }
  if (!all(c("sample_score", "evaluation_metrics") %in% names(data))) {
    stop("'data' must contain 'sample_score' and 'evaluation_metrics'.")
  }
  if (!all(c("label", "score") %in% names(data$sample_score))) {
    stop("'data$sample_score' must contain 'label' and 'score' columns.")
  }

  threshold <- data$evaluation_metrics$Final_Threshold %||% data$evaluation_metrics$`Final_Threshold`
  if (is.null(threshold) || is.na(threshold)) {
    stop("A valid threshold ('Final_Threshold' or 'Final_Threshold') was not found.")
  }

  df <- as.data.frame(data$sample_score)
  df$label <- as.numeric(as.character(df$label))
  df$score <- as.numeric(as.character(df$score))
  df <- df[!is.na(df$label) & !is.na(df$score), ]

  if (nrow(df) == 0) stop("Data is empty after removing NAs.")
  if (length(unique(df$label)) < 2 || !all(unique(df$label) %in% c(0, 1))) {
    stop("'label' column must contain both 0 and 1.")
  }

  plot_obj <- NULL

  if (type == "roc") {
    roc_obj <- pROC::roc(df$label, df$score, quiet = TRUE)
    auc_value <- as.numeric(roc_obj$auc)
    coords_at_threshold <- pROC::coords(roc_obj, x = threshold, ret = c("sensitivity", "specificity"))

    plot_obj <- ggplot(data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities), aes(x = FPR, y = TPR)) +
      geom_line(color = primary_color, linewidth = 1.2) +
      geom_abline(linetype = "dashed", color = "gray50") +
      geom_point(data = data.frame(FPR = 1 - coords_at_threshold$specificity, TPR = coords_at_threshold$sensitivity),
                 color = accent_color, size = 3.5, shape = 18) +
      labs(title = "Receiver Operating Characteristic (ROC) Curve",
           subtitle = paste0("AUC = ", sprintf("%.3f", auc_value)),
           x = "1 - Specificity (False Positive Rate)",
           y = "Sensitivity (True Positive Rate)") +
      theme_bw(base_size = 14) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5), plot.subtitle = element_text(hjust = 0.5),
            panel.grid.minor = element_blank()) +
      coord_fixed()

  } else if (type == "prc") {
    prc_obj <- PRROC::pr.curve(scores.class0 = df$score[df$label == 1], scores.class1 = df$score[df$label == 0], curve = TRUE)
    auprc_value <- prc_obj$auc.integral
    prc_data <- data.frame(recall = prc_obj$curve[, 1], precision = prc_obj$curve[, 2])

    plot_obj <- ggplot(prc_data, aes(x = recall, y = precision)) +
      geom_line(color = secondary_color, linewidth = 1.2) +
      labs(title = "Precision-Recall Curve (PRC)",
           subtitle = paste0("AUPRC = ", sprintf("%.3f", auprc_value)),
           x = "Recall (Sensitivity)", y = "Precision") +
      scale_x_continuous(limits = c(0, 1)) + scale_y_continuous(limits = c(0, 1)) +
      theme_bw(base_size = 14) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5), plot.subtitle = element_text(hjust = 0.5),
            panel.grid.minor = element_blank()) +
      coord_fixed()

  } else if (type == "matrix") {
    predicted_labels <- factor(ifelse(df$score > threshold, 1, 0), levels = c(0, 1))
    actual_labels <- factor(df$label, levels = c(0, 1))
    cm_table <- table(Predicted = predicted_labels, Actual = actual_labels)
    cm_df <- as.data.frame(cm_table)
    cm_df$Percentage <- sprintf("%.1f%%", cm_df$Freq / sum(cm_df$Freq) * 100)

    plot_obj <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile(color = "white", linewidth = 2) +
      geom_text(aes(label = paste0(Freq, "\n(", Percentage, ")")), color = "white", size = 6, fontface = "bold") +
      scale_fill_gradient(low = "#B2DFDB", high = "#00796B", name = "Count") +
      scale_x_discrete(labels = c("Negative (0)", "Positive (1)")) +
      scale_y_discrete(labels = c("Negative (0)", "Positive (1)")) +
      labs(title = "Confusion Matrix", subtitle = paste0("Threshold: ", sprintf("%.3f", threshold)),
           x = "Actual Class", y = "Predicted Class") +
      theme_minimal(base_size = 14) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5), plot.subtitle = element_text(hjust = 0.5),
            axis.text = element_text(face = "bold"), legend.position = "right", panel.grid = element_blank()) +
      coord_fixed()
  }

  if (!is.null(file) && !is.null(plot_obj)) {
    tryCatch({
      ggsave(filename = file, plot = plot_obj, width = 8, height = 8, dpi = 300)
      message(sprintf("Plot saved to: %s", file))
    }, error = function(e) {
      warning(sprintf("Failed to save plot to '%s'. Error: %s", file, e$message))
    })
  }

  return(invisible(plot_obj))
}

#' re-export Surv from survival
#'
#' @importFrom survival Surv
#' @name Surv
#' @export
NULL

# ------------------------------------------------------------------------------
# 2. Prognostic Model Visualization Function (figure_pro)
# ------------------------------------------------------------------------------
#' @title Plot Prognostic Model Evaluation Figures
#' @description Generates and returns a ggplot object for Kaplan-Meier (KM)
#'   survival curves or time-dependent ROC curves.
#'
#' @param type "km" or "tdroc"
#' @param data list with:
#'   - sample_score: data.frame(time, outcome, score)
#'   - evaluation_metrics: for "km" needs KM_Cutoff; for "tdroc" needs AUROC_Years
#'     (numeric years like c(1,3,5), OR a named vector/list like c('1'=0.74,'3'=0.82,'5'=0.85))
#' @param file optional path to save
#' @param time_unit "days" (default), "months", or "years" for df$time
#'
#' @return ggplot object
#' @export
figure_pro <- function(type, data, file = NULL, time_unit = "days") {

  if (!type %in% c("km", "tdroc")) stop("Invalid 'type'. Choose 'km' or 'tdroc'.")
  if (!all(c("sample_score", "evaluation_metrics") %in% names(data))) {
    stop("'data' must contain 'sample_score' and 'evaluation_metrics'.")
  }
  if (!all(c("time", "outcome", "score") %in% names(data$sample_score))) {
    stop("'data$sample_score' must contain 'time', 'outcome', and 'score'.")
  }

  df <- as.data.frame(data$sample_score)
  df$time    <- as.numeric(df$time)
  df$outcome <- as.numeric(df$outcome)
  df$score   <- as.numeric(df$score)
  df <- df[stats::complete.cases(df[, c("time","outcome","score")]), ]

  if (nrow(df) == 0) stop("Data is empty after removing NAs.")
  if (length(unique(df$outcome)) < 2 || !all(unique(df$outcome) %in% c(0, 1))) {
    stop("'outcome' column must contain both 0 and 1.")
  }

  # ---------- helpers ----------
  .time_factor <- function(unit) {
    switch(tolower(unit),
           "days"   = 365.25,   # years -> days
           "months" = 12,       # years -> months
           "years"  = 1,
           1)
  }

  .normalize_years <- function(x) {
    # numeric vector of years
    if (is.numeric(x) && all(x >= 0)) return(sort(unique(as.numeric(x))))

    # named vector/list where names are the years (values often AUCs)
    nms <- names(x)
    if (!is.null(nms)) {
      yrs <- suppressWarnings(as.numeric(nms))
      if (all(!is.na(yrs))) return(sort(unique(yrs)))
    }

    # vector/list of AUCs without names -> fallback
    vals <- suppressWarnings(as.numeric(unlist(x, use.names = FALSE)))
    if (all(!is.na(vals)) && all(vals > 0 & vals < 1)) {
      warning("'AUROC_Years' looks like AUC values without year names. Falling back to c(1,3,5).")
      return(c(1,3,5))
    }

    stop("`AUROC_Years` must be numeric years (e.g., c(1,3,5)) OR a named list/vector where names are the years.")
  }

  # ------------- plotting -------------
  if (type == "km") {
    cutoff <- data$evaluation_metrics$KM_Cutoff
    if (is.null(cutoff)) stop("'KM_Cutoff' is missing from data$evaluation_metrics.")

    df$risk_group <- factor(ifelse(df$score > cutoff, "High Risk", "Low Risk"),
                            levels = c("Low Risk", "High Risk"))
    if (length(unique(df$risk_group)) < 2) {
      warning("Only one risk group present after applying cutoff. KM plot may not be meaningful.")
    }

    fit <- survival::survfit(survival::Surv(time, outcome) ~ risk_group, data = df)
    km_list <- survminer::ggsurvplot(
      fit, data = df, pval = TRUE, conf.int = TRUE, risk.table = TRUE,
      xlab = paste0("Time (", time_unit, ")"), ylab = "Overall Survival Probability",
      title = "Kaplan-Meier Survival Curve", legend.title = "Risk Group",
      palette = c("#2E86AB", "#A23B72"), ggtheme = ggplot2::theme_bw(base_size = 14)
    )
    plot_obj <- km_list$plot

  } else { # type == "tdroc"
    raw_eval <- data$evaluation_metrics$AUROC_Years
    if (is.null(raw_eval)) stop("'AUROC_Years' is missing from data$evaluation_metrics.")
    eval_years <- .normalize_years(raw_eval)

    pre_auc <- NULL
    if (!is.null(names(raw_eval))) {
      pre_auc <- as.numeric(unlist(raw_eval))
      names(pre_auc) <- names(raw_eval)
    }

    factor <- .time_factor(time_unit)
    eval_times <- eval_years * factor

    # Attempt using the more robust timeROC package first
    roc_res <- tryCatch({
      timeROC::timeROC(T = df$time, delta = df$outcome, marker = df$score,
                       cause = 1, times = eval_times, iid = FALSE)
    }, error = function(e) NULL)

    roc_df_list <- list()
    for (i in seq_along(eval_years)) {
      yr  <- eval_years[i]
      tpt <- eval_times[i]

      FPR <- NULL; TPR <- NULL; auc_calc <- NA_real_

      # Primary method: Extract results from timeROC
      use_timeROC <- !is.null(roc_res) && i <= NCOL(roc_res$FP) && !all(is.na(roc_res$FP[, i]))
      if (use_timeROC) {
        FPR <- roc_res$FP[, i]
        TPR <- roc_res$TP[, i]
        auc_calc <- roc_res$AUC[i]
      } else {
        # Fallback method: Use survivalROC for individual time points
        sroc <- tryCatch({
          survivalROC::survivalROC(Stime = df$time, status = df$outcome,
                                   marker = df$score, predict.time = tpt,
                                   method = "NNE", span = 0.25)
        }, error = function(e) NULL)

        if (!is.null(sroc) && !all(is.na(sroc$FP)) && !all(is.na(sroc$TP))) {
          FPR <- sroc$FP
          TPR <- sroc$TP
          auc_calc <- sroc$AUC
        }
      }

      # ***MODIFICATION***: Check if calculation was successful, otherwise warn and skip.
      if (is.null(FPR) || is.null(TPR)) {
        warning(sprintf("Could not compute ROC curve for %d-Year time point. Skipping. (This may be due to insufficient events before this time).", yr))
        next
      }

      # Prioritize pre-calculated AUC for the label, otherwise use the calculated one
      auc_for_label <- if (!is.null(pre_auc) && as.character(yr) %in% names(pre_auc)) {
        suppressWarnings(as.numeric(pre_auc[as.character(yr)]))
      } else {
        auc_calc
      }

      # ***MODIFICATION***: Create a more robust label
      auc_text <- ifelse(is.na(auc_for_label), "N/A", sprintf("%.3f", auc_for_label))
      label_text <- sprintf("%d-Year (AUC=%s)", as.integer(yr), auc_text)

      roc_df_list[[as.character(yr)]] <- data.frame(FPR = FPR, TPR = TPR, Label = label_text, stringsAsFactors = FALSE)
    }

    if (length(roc_df_list) == 0)
      stop("Failed to compute any time-dependent ROC curves for the requested years.")

    all_roc_data <- do.call(rbind, roc_df_list)

    # ***MODIFICATION***: Create ordered factor for the legend to ensure correct order
    year_order <- sort(as.numeric(names(roc_df_list)))
    ordered_labels <- sapply(year_order, function(y) roc_df_list[[as.character(y)]]$Label[1])
    all_roc_data$Label <- factor(all_roc_data$Label, levels = ordered_labels)

    plot_obj <- ggplot2::ggplot(all_roc_data, ggplot2::aes(x = FPR, y = TPR, color = Label)) +
      ggplot2::geom_line(linewidth = 1.1) +
      ggplot2::geom_abline(linetype = "dashed", color = "gray50") +
      ggplot2::labs(
        title = "Time-Dependent ROC Curves",
        x = "1 - Specificity",
        y = "Sensitivity",
        color = "Time Point"
      ) +
      ggplot2::theme_bw(base_size = 14) +
      ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", hjust = 0.5),
                     legend.position = "bottom") +
      ggplot2::coord_fixed()
  }

  if (!is.null(file) && !is.null(plot_obj)) {
    tryCatch({
      ggplot2::ggsave(filename = file, plot = plot_obj, width = 8, height = 8, dpi = 300)
      message(sprintf("Plot saved to: %s", file))
    }, error = function(e) {
      warning(sprintf("Failed to save plot to '%s'. Error: %s", file, e$message))
    })
  }

  invisible(plot_obj)
}

# ------------------------------------------------------------------------------
# 3. SHAP Model Explanation Function (figure_shap)
# ------------------------------------------------------------------------------
#' @title Generate and Plot SHAP Explanation Figures
#' @description Creates SHAP (SHapley Additive exPlanations) plots to explain
#'   feature contributions by training a surrogate model on the original model's scores.
#'
#' @param data A list containing `sample_score`, a data frame with sample IDs and `score`.
#' @param raw_data A data frame with original features. The first column must be the sample ID.
#' @param target_type String, the analysis type: "diagnosis" or "prognosis".
#'   This determines which columns in `raw_data` are treated as features.
#' @param file Optional. A string specifying the path to save the plot. If `NULL`
#'   (default), the plot object is returned.
#' @param model_type String, the surrogate model for SHAP calculation.
#'   "xgboost" (default) or "lasso".
#'
#' @return A patchwork object combining SHAP summary and importance plots. If `file` is
#'   provided, the plot is also saved.
#' @examples
#' \donttest{
#' # --- Example for a Diagnosis Model ---
#' set.seed(123)
#' train_dia_data <- data.frame(
#'   SampleID = paste0("S", 1:100),
#'   Label = sample(c(0, 1), 100, replace = TRUE),
#'   FeatureA = rnorm(100, 10, 2),
#'   FeatureB = runif(100, 0, 5)
#' )
#' model_results <- list(
#'   sample_score = data.frame(ID = paste0("S", 1:100), score = runif(100, 0, 1))
#' )
#'
#' # Generate SHAP plot object
#' shap_plot <- figure_shap(
#'   data = model_results,
#'   raw_data = train_dia_data,
#'   target_type = "diagnosis",
#'   model_type = "xgboost"
#' )
#' # To display the plot:
#' # print(shap_plot)
#' }
#' @importFrom dplyr inner_join select
#' @importFrom xgboost xgb.DMatrix xgb.train
#' @importFrom glmnet cv.glmnet
#' @importFrom shapviz shapviz sv_importance
#' @importFrom patchwork plot_layout
#' @importFrom stats reorder complete.cases sd
#' @importFrom utils head
#' @importFrom ggplot2 theme
#' @export
figure_shap <- function(data, raw_data, target_type, file = NULL, model_type = "xgboost") {

  target_type <- match.arg(target_type, c("diagnosis", "prognosis"))

  if (!"sample_score" %in% names(data) || !"score" %in% names(data$sample_score)) {
    stop("'data' must be a list containing 'sample_score' with a 'score' column.")
  }

  score_df <- data$sample_score
  names(score_df)[1] <- "ID"
  names(raw_data)[1] <- "ID"

  merged_df <- dplyr::inner_join(raw_data, score_df, by = "ID")
  merged_df <- merged_df[!is.na(merged_df$score), ]
  if (nrow(merged_df) == 0) stop("No matching samples with non-NA scores found.")

  feature_start_col <- if (target_type == "diagnosis") 3 else 4
  if (ncol(raw_data) < feature_start_col) stop("Not enough columns in 'raw_data' for the selected 'target_type'.")

  feature_cols <- names(raw_data)[-c(1:(feature_start_col - 1))]
  X_features <- merged_df[, feature_cols, drop = FALSE]
  X_features <- data.matrix(X_features)
  target_score <- merged_df$score

  if (any(!stats::complete.cases(X_features))) {
    warning("NA values found in features; rows with NAs will be removed for SHAP analysis.")
    complete_idx <- stats::complete.cases(X_features)
    X_features <- X_features[complete_idx, , drop = FALSE]
    target_score <- target_score[complete_idx]
  }
  if (nrow(X_features) == 0) stop("Feature data is empty after removing NAs.")

  message(sprintf("Training '%s' surrogate model and calculating SHAP values...", model_type))
  surrogate_model <- NULL
  if (model_type == "xgboost") {
    dtrain <- xgboost::xgb.DMatrix(X_features, label = target_score)
    surrogate_model <- xgboost::xgb.train(params = list(objective = "reg:squarederror", nthread = 1), data = dtrain, nrounds = 100)
  } else if (model_type == "lasso") {
    surrogate_model <- glmnet::cv.glmnet(X_features, target_score, alpha = 1, family = "gaussian")
  }
  if (is.null(surrogate_model)) stop("Surrogate model training failed.")

  sv <- shapviz::shapviz(surrogate_model, X_pred = X_features)

  p_beeswarm <- shapviz::sv_importance(sv, kind = "beeswarm", max_display = 15) +
    labs(title = "SHAP Summary Plot", x = "SHAP value (impact on model score)") +
    theme_minimal(base_size = 14) + theme(plot.title = element_text(face = "bold", hjust = 0.5))

  p_bar <- shapviz::sv_importance(sv, kind = "bar") +
    labs(title = "Feature Importance", subtitle = "Mean Absolute SHAP Value", x = NULL, y = NULL) +
    theme_minimal(base_size = 14) + theme(plot.title = element_text(face = "bold", hjust = 0.5))

  combined_plot <- p_beeswarm + p_bar + patchwork::plot_layout(ncol = 1, heights = c(2, 1.5))

  if (!is.null(file)) {
    tryCatch({
      ggsave(filename = file, plot = combined_plot, width = 10, height = 12, dpi = 300)
      message(sprintf("SHAP plot saved to: %s", file))
    }, error = function(e) {
      warning(sprintf("Failed to save SHAP plot to '%s'. Error: %s", file, e$message))
    })
  }

  return(invisible(combined_plot))
}

# Helper for providing default values (equivalent to Python's .get(key, default))
`%||%` <- function(a, b) {
  if (is.null(a)) b else a
}
