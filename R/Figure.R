# Figure.R

# Global Aesthetic Color Settings
# These colors are used consistently across plotting functions for branding and clarity.
primary_color <- "#2E86AB"   # Deep blue, often used for primary lines or fills.
secondary_color <- "#A23B72" # Magenta, used for secondary elements or contrasts.
accent_color <- "#F18F01"    # Orange, used for highlighting specific points or annotations.

#' @importFrom utils globalVariables
utils::globalVariables(c("FPR", "TPR", "TimePoint"))

# ------------------------------------------------------------------------------
# 2. Diagnostic Model Visualization Function (figure_dia)
# ------------------------------------------------------------------------------

#' @title Plot Diagnostic Model Evaluation Figures
#' @description Generates Receiver Operating Characteristic (ROC) curves,
#'   Precision-Recall (PRC) curves, or confusion matrices based on diagnostic model
#'   evaluation results.
#'
#' @param type String, specifies the type of plot to generate. Options are
#'   "roc" (ROC curve), "prc" (Precision-Recall curve), or "matrix" (Confusion Matrix).
#' @param data A list object containing model evaluation results. It must include:
#'   \itemize{
#'     \item `sample_score`: A data frame with at least "ID", "label" (0/1), and "score" columns.
#'     \item `evaluation_metrics`: A list with a "Final_Threshold" or "_Threshold" value.
#'   }
#' @param output_file String, the base name for the output file (without extension).
#' @param output_type String, the desired output file format. Options: "pdf", "png", "svg".
#'   Defaults to "pdf".
#'
#' @return NULL. The function saves the generated plot directly to a file.
#' @examples
#' \dontrun{
#' # Example data structure for diagnostic model evaluation results:
#' external_eval_example_dia <- list(
#'   sample_score = data.frame(
#'     ID = paste0("S", 1:100),
#'     label = sample(c(0, 1), 100, replace = TRUE),
#'     score = runif(100, 0, 1)
#'   ),
#'   evaluation_metrics = list(
#'     Final_Threshold = 0.53,
#'     AUROC = 0.75,
#'     AUPRC = 0.68
#'   )
#' )
#'
#' # Plot ROC curve
#' figure_dia(type = "roc", data = external_eval_example_dia,
#'            output_file = "Diagnostic_Model_ROC", output_type = "png")
#'
#' # Plot PRC curve
#' figure_dia(type = "prc", data = external_eval_example_dia,
#'            output_file = "Diagnostic_Model_PRC", output_type = "png")
#'
#' # Plot Confusion Matrix
#' figure_dia(type = "matrix", data = external_eval_example_dia,
#'            output_file = "Diagnostic_Model_Matrix", output_type = "png")
#' }
#' @importFrom pROC roc coords
#' @importFrom PRROC pr.curve
#' @importFrom ggplot2 ggplot aes geom_line geom_abline geom_point annotate labs
#'   scale_x_continuous scale_y_continuous theme_bw element_text element_blank
#'   geom_tile geom_text scale_fill_gradient scale_x_discrete scale_y_discrete
#'   theme_minimal coord_fixed ggsave
#' @importFrom dplyr select
#' @importFrom Cairo CairoPDF CairoPNG CairoSVG
#' @export
figure_dia <- function(type, data, output_file, output_type = "pdf") {

  if (!type %in% c("roc", "prc", "matrix")) {
    stop("Invalid 'type' parameter. Diagnostic model currently supports 'roc', 'prc', or 'matrix'.")
  }
  if (!all(c("sample_score", "evaluation_metrics") %in% names(data))) {
    stop("'data' object format is incorrect; it must contain 'sample_score' and 'evaluation_metrics'.")
  }
  if (!"score" %in% names(data$sample_score)) {
    stop("'data$sample_score' must contain a 'score' column.")
  }

  threshold <- data$evaluation_metrics$Final_Threshold
  if (is.null(threshold) || is.na(threshold)) {
    threshold <- data$evaluation_metrics$`_Threshold`
    if (is.null(threshold) || is.na(threshold)) {
      stop("No valid threshold found. Ensure 'data$evaluation_metrics' contains 'Final_Threshold' or '_Threshold'.")
    }
  }

  df <- as.data.frame(data$sample_score)

  initial_rows <- nrow(df)
  tryCatch({
    df$score <- as.numeric(as.character(df$score))
  }, error = function(e) {
    stop(paste0("Failed to convert diagnostic model score column to numeric: ", e$message))
  })

  df <- df[!is.na(df$score), ]
  if (nrow(df) == 0) {
    stop("Diagnostic dataframe is empty after removing NA values, cannot visualize.")
  }
  if (nrow(df) < initial_rows) {
    warning(sprintf("Removed %d rows from diagnostic data due to NA values for visualization.", initial_rows - nrow(df)))
  }

  if (type %in% c("roc", "prc", "matrix")) {
    if (!"label" %in% names(df)) {
      stop(paste0("Plotting ", type, " requires a 'label' column."))
    }

    tryCatch({
      df$label <- as.numeric(as.character(df$label))
    }, error = function(e) {
      stop(paste0("Failed to convert diagnostic label column to numeric: ", e$message))
    })

    df <- df[!is.na(df$label), ]
    if (nrow(df) == 0) {
      stop("Diagnostic dataframe is empty after removing NA label values, cannot visualize.")
    }

    unique_labels <- unique(df$label)
    if (!all(unique_labels %in% c(0, 1)) || length(unique_labels) < 2) {
      stop("Diagnostic labels (label) must exclusively contain 0 and 1, and both must be present for meaningful plots.")
    }
  }

  output_filename <- paste0(output_file, ".", output_type)
  plot_obj <- NULL

  if (type == "roc") {
    roc_obj <- pROC::roc(df$label, df$score, quiet = TRUE)
    auc_value <- as.numeric(roc_obj$auc)

    roc_data <- data.frame(
      specificity = roc_obj$specificities,
      sensitivity = roc_obj$sensitivities
    )

    coords_at_threshold <- pROC::coords(roc_obj, x = threshold, input = "threshold",
                                        ret = c("sensitivity", "specificity"))

    plot_obj <- ggplot2::ggplot(roc_data, ggplot2::aes(x = 1 - specificity, y = sensitivity)) +
      ggplot2::geom_line(color = primary_color, linewidth = 1.2) +
      ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
      ggplot2::geom_point(data = data.frame(x = 1 - coords_at_threshold$specificity,
                                            y = coords_at_threshold$sensitivity),
                          ggplot2::aes(x = x, y = y), color = accent_color, size = 3) +
      ggplot2::annotate("text", x = 1 - coords_at_threshold$specificity + 0.1,
                        y = coords_at_threshold$sensitivity - 0.05,
                        label = paste0("Threshold: ", round(threshold, 3)),
                        color = accent_color, fontface = "bold") +
      ggplot2::scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
      ggplot2::scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
      ggplot2::labs(
        title = "Receiver Operating Characteristic (ROC) Curve",
        subtitle = paste0("AUC = ", sprintf("%.3f", auc_value)),
        x = "1 - Specificity (False Positive Rate)",
        y = "Sensitivity (True Positive Rate)"
      ) +
      ggplot2::theme_bw() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(size = 18, face = "bold", hjust = 0.5),
        plot.subtitle = ggplot2::element_text(size = 14, hjust = 0.5),
        axis.title = ggplot2::element_text(size = 14, face = "bold"),
        axis.text = ggplot2::element_text(size = 12),
        panel.grid.minor = ggplot2::element_blank()
      ) +
      ggplot2::coord_fixed()

  } else if (type == "prc") {
    prc_obj <- PRROC::pr.curve(scores.class0 = df$score[df$label == 1],
                               scores.class1 = df$score[df$label == 0],
                               curve = TRUE)
    auprc_value <- prc_obj$auc.integral

    prc_data <- data.frame(
      recall = prc_obj$curve[, 1],
      precision = prc_obj$curve[, 2]
    )

    predicted_labels <- base::ifelse(df$score > threshold, 1, 0)
    tp <- sum(predicted_labels == 1 & df$label == 1)
    fp <- sum(predicted_labels == 1 & df$label == 0)
    fn <- sum(predicted_labels == 0 & df$label == 1)

    precision_at_threshold <- base::ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall_at_threshold <- base::ifelse(tp + fn > 0, tp / (tp + fn), 0)

    plot_obj <- ggplot2::ggplot(prc_data, ggplot2::aes(x = recall, y = precision)) +
      ggplot2::geom_line(color = secondary_color, linewidth = 1.2) +
      ggplot2::geom_point(data = data.frame(x = recall_at_threshold, y = precision_at_threshold),
                          ggplot2::aes(x = x, y = y), color = accent_color, size = 3) +
      ggplot2::annotate("text", x = recall_at_threshold - 0.1, y = precision_at_threshold + 0.05,
                        label = paste0("Threshold: ", round(threshold, 3)),
                        color = accent_color, fontface = "bold") +
      ggplot2::scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
      ggplot2::scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
      ggplot2::labs(
        title = "Precision-Recall Curve (PRC)",
        subtitle = paste0("AUPRC = ", sprintf("%.3f", auprc_value)),
        x = "Recall (Sensitivity)",
        y = "Precision"
      ) +
      ggplot2::theme_bw() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(size = 18, face = "bold", hjust = 0.5),
        plot.subtitle = ggplot2::element_text(size = 14, hjust = 0.5),
        axis.title = ggplot2::element_text(size = 14, face = "bold"),
        axis.text = ggplot2::element_text(size = 12),
        panel.grid.minor = ggplot2::element_blank()
      ) +
      ggplot2::coord_fixed()

  } else if (type == "matrix") {
    predicted_labels <- base::ifelse(df$score > threshold, 1, 0)

    cm_table <- table(Predicted = factor(predicted_labels, levels = c(0, 1)),
                      Actual = factor(df$label, levels = c(0, 1)))

    cm_df <- as.data.frame(cm_table)

    cm_df$Percentage <- cm_df$Freq / sum(cm_df$Freq) * 100

    tn <- cm_table[1, 1]; fp <- cm_table[2, 1]
    fn <- cm_table[1, 2]; tp <- cm_table[2, 2]

    accuracy <- (tp + tn) / (tp + tn + fp + fn)
    sensitivity <- tp / (tp + fn)
    specificity <- tn / (tn + fp)
    precision <- tp / (tp + fp)

    plot_obj <- ggplot2::ggplot(cm_df, ggplot2::aes(x = Actual, y = Predicted, fill = Freq)) +
      ggplot2::geom_tile(color = "white", linewidth = 2) +
      ggplot2::geom_text(ggplot2::aes(label = paste0(Freq, "\n(", sprintf("%.1f", Percentage), "%)")),
                         color = "white", size = 6, fontface = "bold") +
      ggplot2::scale_fill_gradient(low = "lightblue", high = "darkblue", name = "Count") +
      ggplot2::scale_x_discrete(labels = c("Negative", "Positive")) +
      ggplot2::scale_y_discrete(labels = c("Negative", "Positive")) +
      ggplot2::labs(
        title = "Confusion Matrix",
        subtitle = paste0("Threshold: ", sprintf("%.3f", threshold),
                          " | Accuracy: ", sprintf("%.3f", accuracy),
                          " | Sensitivity: ", sprintf("%.3f", sensitivity),
                          " | Specificity: ", sprintf("%.3f", specificity)),
        x = "Actual Class",
        y = "Predicted Class"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(size = 18, face = "bold", hjust = 0.5),
        plot.subtitle = ggplot2::element_text(size = 12, hjust = 0.5),
        axis.title = ggplot2::element_text(size = 14, face = "bold"),
        axis.text = ggplot2::element_text(size = 12, face = "bold"),
        legend.position = "right",
        panel.grid = ggplot2::element_blank()
      ) +
      ggplot2::coord_fixed()
  }

  if (!is.null(plot_obj)) {
    if (output_type == "pdf") {
      Cairo::CairoPDF(output_filename, width = 8, height = 8)
    } else if (output_type == "png") {
      Cairo::CairoPNG(output_filename, width = 8 * 300, height = 8 * 300, res = 300)
    } else if (output_type == "svg") {
      Cairo::CairoSVG(output_filename, width = 8, height = 8)
    } else {
      warning(paste0("Unsupported output type: ", output_type, ". Attempting ggsave fallback."))
      ggplot2::ggsave(output_filename, plot = plot_obj, device = output_type,
                      width = 8, height = 8, dpi = 300)
      message(sprintf("'%s' plot saved to: %s", type, output_filename))
      return(invisible(NULL))
    }
    print(plot_obj)
    grDevices::dev.off()
    message(sprintf("'%s' plot saved to: %s", type, output_filename))
  } else {
    warning("No plot object generated, file not saved.")
  }
}


# ------------------------------------------------------------------------------
# 3. Prognostic Model Visualization Function (figure_pro) - REVISED with timeROC
# ------------------------------------------------------------------------------
#' @title Plot Prognostic Model Evaluation Figures
#' @description Generates Kaplan-Meier (KM) survival curves or time-dependent ROC curves
#'   based on prognostic model evaluation results. This version uses the 'timeROC'
#'   package for more robust and efficient tdROC calculations.
#'
#' @param type String, specifies the type of plot to generate. Options are "km"
#'   (Kaplan-Meier curve) or "tdroc" (Time-Dependent ROC curve). For "tdroc", multiple
#'   ROC curves for specified time points are overlaid on a single plot.
#' @param data A list object containing model evaluation results. It must include:
#'   \itemize{
#'     \item `sample_score`: A data frame with at least "ID", "time", "outcome" (0/1), and "score" columns.
#'     \item `evaluation_metrics`: A list with a "KM_Cutoff" value for KM plots, and
#'       "AUROC_Years" (a numeric vector or list of evaluation years) for tdROC plots.
#'   }
#' @param output_file String, the base name for the output file (without extension).
#' @param output_type String, the desired output file format. Options: "pdf", "png", "svg".
#'   Defaults to "pdf".
#' @param time_unit String, specifies the unit of time for the `time` column in `sample_score`
#'   (e.g., "days", "months", "years"). This is crucial for correctly calculating tdROC.
#'   Defaults to "days".
#'
#' @return NULL. The function saves the generated plot directly to a file.
#' @examples
#' \dontrun{
#' # Ensure required packages are installed
#' # install.packages(c("survival", "survminer", "timeROC", "ggplot2", "Cairo"))
#'
#' # Example data structure for prognostic model evaluation results:
#' set.seed(42)
#' external_eval_example_pro <- list(
#'   sample_score = data.frame(
#'     ID = paste0("S", 1:200),
#'     time = runif(200, 10, 1825), # time in days
#'     outcome = sample(c(0, 1), 200, replace = TRUE, prob = c(0.7, 0.3)),
#'     score = runif(200, 0, 1)
#'   ),
#'   evaluation_metrics = list(
#'     KM_Cutoff = 0.5,
#'     AUROC_Years = c(1, 3, 5) # Evaluation years
#'   )
#' )
#'
#' # Plot Kaplan-Meier curve
#' figure_pro(type = "km", data = external_eval_example_pro,
#'            output_file = "Prognostic_Model_KM", output_type = "png",
#'            time_unit = "days")
#'
#' # Plot Time-Dependent ROC curves (1-year, 3-year, 5-year overlaid)
#' figure_pro(type = "tdroc", data = external_eval_example_pro,
#'            output_file = "Prognostic_Model_TDROC", output_type = "png",
#'            time_unit = "days")
#' }
#' @importFrom survival Surv survfit
#' @importFrom survminer ggsurvplot
#' @importFrom timeROC timeROC
#' @importFrom ggplot2 ggplot aes geom_line geom_abline labs scale_x_continuous
#'   scale_y_continuous theme_bw element_text element_blank ggsave scale_color_manual
#'   coord_fixed
#' @importFrom Cairo CairoPDF CairoPNG CairoSVG
#' @importFrom grDevices dev.off
#' @export
figure_pro <- function(type, data, output_file, output_type = "pdf", time_unit = "days") {

  required_pkgs <- c("survival", "ggplot2", "Cairo")
  if (type == "km") {
    required_pkgs <- c(required_pkgs, "survminer")
  } else if (type == "tdroc") {
    required_pkgs <- c(required_pkgs, "timeROC")
  }

  for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(paste("Package '", pkg, "' is required but not installed. Please install it with install.packages('", pkg, "')."))
    }
    # Load the package into the session
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }

  # --- Input Validation and Data Preparation (Unchanged) ---
  if (!type %in% c("km", "tdroc")) {
    stop("Invalid 'type'. Prognostic model supports 'km' or 'tdroc'.")
  }
  if (!all(c("sample_score", "evaluation_metrics") %in% names(data))) {
    stop("'data' must contain 'sample_score' and 'evaluation_metrics'.")
  }
  if (!all(c("time", "outcome", "score") %in% names(data$sample_score))) {
    stop("'data$sample_score' must contain 'time', 'outcome', and 'score'.")
  }

  df <- as.data.frame(data$sample_score)

  initial_rows <- nrow(df)
  tryCatch({
    df$time <- as.numeric(as.character(df$time))
    df$outcome <- as.numeric(as.character(df$outcome))
    df$score <- as.numeric(as.character(df$score))
  }, error = function(e) {
    stop(paste0("Failed to convert prognostic columns to numeric. Error: ", e$message))
  })

  df <- df[!is.na(df$time) & !is.na(df$outcome) & !is.na(df$score), ]
  if (nrow(df) == 0) {
    stop("Prognostic dataframe is empty after removing NA values.")
  }
  if (nrow(df) < initial_rows) {
    warning(sprintf("Removed %d rows due to NA values.", initial_rows - nrow(df)))
  }

  invalid_outcome_rows <- !df$outcome %in% c(0, 1)
  if (any(invalid_outcome_rows)) {
    warning(sprintf("Found and removed %d rows with outcome values other than 0 or 1.", sum(invalid_outcome_rows)))
    df <- df[!invalid_outcome_rows, ]
  }
  if (length(unique(df$outcome)) < 2) {
    stop("Prognostic outcome must contain both 0 and 1 for meaningful plots.")
  }

  output_filename <- paste0(output_file, ".", output_type)
  plot_obj <- NULL

  # --- Plot Generation ---

  if (type == "km") {
    cutoff <- data$evaluation_metrics$KM_Cutoff
    if (is.null(cutoff) || is.na(cutoff)) {
      stop("Cannot plot KM curve; 'KM_Cutoff' is invalid in data$evaluation_metrics.")
    }
    df$risk_group <- base::ifelse(df$score > cutoff, "High Risk", "Low Risk")
    df$risk_group <- factor(df$risk_group, levels = c("Low Risk", "High Risk"))

    # Check if there are events in both groups
    if(length(unique(df$risk_group)) < 2){
      warning("Only one risk group present after applying cutoff. KM plot may not be meaningful.")
    }

    fit <- survival::survfit(survival::Surv(time, outcome) ~ risk_group, data = df)
    plot_obj <- survminer::ggsurvplot(
      fit, data = df, pval = TRUE, conf.int = TRUE, risk.table = TRUE,
      risk.table.col = "strata", risk.table.y.text = FALSE,
      xlab = paste0("Time (", time_unit, ")"), ylab = "Overall Survival Probability",
      title = "Kaplan-Meier Survival Curve", legend.title = "Risk Group",
      legend.labs = c("Low Risk", "High Risk"), palette = c("#0073C2FF", "#E7B800FF"),
      ggtheme = ggplot2::theme_bw()
    )

  } else if (type == "tdroc") {
    # --- TD-ROC Plotting using the 'timeROC' package ---
    if (!"AUROC_Years" %in% names(data$evaluation_metrics)) {
      stop("Plotting tdROC requires 'AUROC_Years' in data$evaluation_metrics.")
    }
    auroc_obj <- data$evaluation_metrics$AUROC_Years
    eval_time_points_yr <- NULL
    if (is.list(auroc_obj) && !is.null(names(auroc_obj))) {
      eval_time_points_yr <- as.numeric(names(auroc_obj))
    } else if (is.numeric(auroc_obj)) {
      eval_time_points_yr <- auroc_obj
    }
    if (is.null(eval_time_points_yr) || any(is.na(eval_time_points_yr)) || length(eval_time_points_yr) == 0) {
      stop("Could not find valid numeric evaluation time points in 'AUROC_Years'.")
    }
    eval_time_points_yr <- sort(unique(eval_time_points_yr))

    time_conversion_factor <- switch(time_unit,
                                     "days" = 365.25,
                                     "months" = 12,
                                     "years" = 1,
                                     1)
    if (!time_unit %in% c("days", "months", "years")) {
      warning(paste("Unrecognized 'time_unit':", time_unit, ". Assuming it's the same scale as AUROC_Years (years)."))
    }
    eval_time_points_converted <- eval_time_points_yr * time_conversion_factor

    # Calculate ROC using timeROC - more efficient and stable
    roc_result <- tryCatch({
      timeROC::timeROC(
        T = df$time,
        delta = df$outcome,
        marker = df$score,
        cause = 1, # Define the event of interest as '1'
        times = eval_time_points_converted,
        iid = FALSE # Set to FALSE for speed, as we don't need confidence intervals here
      )
    }, error = function(e) {
      warning(paste("Failed to calculate time-dependent ROC. Error:", e$message))
      return(NULL)
    })

    if (is.null(roc_result) || length(roc_result$TP) == 0) {
      stop("Failed to generate ROC data. Check if there are enough events before the earliest time point.")
    }

    roc_data_list <- list()
    # The number of columns in FP/TP corresponds to the number of time points
    for (i in 1:ncol(roc_result$TP)) {
      time_point_yr <- eval_time_points_yr[i]
      auc_val <- roc_result$AUC[i]

      # Skip if AUC is NA (can happen if no events before time point)
      if(is.na(auc_val)){
        warning(sprintf("Skipping %.1f-year ROC: AUC could not be computed (likely no events before this time).", time_point_yr))
        next
      }

      # Extract False Positive and True Positive rates
      fp <- roc_result$FP[, i]
      tp <- roc_result$TP[, i]

      # Downsample if too many points to prevent rendering issues
      max_points <- 1000
      if (length(fp) > max_points) {
        # Create an evenly spaced sequence of indices to select points
        # This preserves the overall shape of the curve while reducing point density
        seq_idx <- round(seq(from = 1, to = length(fp), length.out = max_points))
        fp <- fp[seq_idx]
        tp <- tp[seq_idx]
      }

      roc_data_list[[as.character(time_point_yr)]] <- data.frame(
        FPR = fp,
        TPR = tp,
        TimePoint = as.character(time_point_yr),
        AUC = auc_val
      )
    }

    if (length(roc_data_list) == 0) {
      stop("Failed to calculate ROC data for any specified time points. Check data and time points.")
    }

    all_roc_data <- do.call(rbind, roc_data_list)
    # Ensure TimePoint is a factor with the correct order for plotting and legend
    all_roc_data$TimePoint <- factor(all_roc_data$TimePoint, levels = as.character(eval_time_points_yr))

    # Create labels for the legend
    legend_labels_df <- unique(all_roc_data[, c("TimePoint", "AUC")])
    legend_labels_df <- legend_labels_df[order(as.numeric(as.character(legend_labels_df$TimePoint))), ]
    legend_labels <- sprintf("%s-Year (AUC = %.3f)", legend_labels_df$TimePoint, legend_labels_df$AUC)

    # Define a color palette
    roc_palette <- c("#E64B35", "#3C5488", "#F39B7F", "#8491B4", "#4DBBD5", "#00A087")
    if (length(eval_time_points_yr) > length(roc_palette)) {
      warning("More time points than defined colors; colors will be recycled.")
      roc_palette <- rep(roc_palette, length.out = length(eval_time_points_yr))
    }

    # Generate the plot object using ggplot2
    plot_obj <- ggplot2::ggplot(all_roc_data, ggplot2::aes(x = FPR, y = TPR, color = TimePoint)) +
      ggplot2::geom_line(linewidth = 1.1) +
      ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
      ggplot2::scale_color_manual(
        name = "Time Point",
        values = roc_palette[1:length(legend_labels)], # Use only as many colors as needed
        labels = legend_labels
      ) +
      ggplot2::scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0.01, 0.01)) +
      ggplot2::scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0.01, 0.01)) +
      ggplot2::labs(
        title = "Time-Dependent ROC Curves",
        x = "1 - Specificity (False Positive Rate)",
        y = "Sensitivity (True Positive Rate)"
      ) +
      ggplot2::theme_bw() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(size = 16, face = "bold", hjust = 0.5),
        axis.title = ggplot2::element_text(size = 14),
        axis.text = ggplot2::element_text(size = 12),
        legend.title = ggplot2::element_text(size = 12, face = "bold"),
        legend.text = ggplot2::element_text(size = 11),
        panel.grid.minor = ggplot2::element_blank(),
        aspect.ratio = 1 # Enforce a square plot
      ) +
      ggplot2::coord_fixed()
  }

  # --- Save Plot (Unchanged) ---
  if (!is.null(plot_obj)) {
    save_width <- 8
    save_height <- 8

    # Use a consistent saving mechanism
    save_plot <- function(filename, plot, type, w, h) {
      if (type == "pdf") {
        Cairo::CairoPDF(filename, width = w, height = h)
        print(plot)
      } else if (type == "png") {
        # For ggsurvplot which is a list, we need to print the plot component
        if(inherits(plot, "ggsurvplot")) plot <- plot$plot
        Cairo::CairoPNG(filename, width = w * 300, height = h * 300, res = 300, units = "px")
        print(plot)
      } else if (type == "svg") {
        if(inherits(plot, "ggsurvplot")) plot <- plot$plot
        Cairo::CairoSVG(filename, width = w, height = h)
        print(plot)
      } else {
        # Fallback for other types
        ggplot2::ggsave(filename, plot = plot, device = type, width = w, height = h, dpi = 300)
        return() # ggsave prints its own message
      }
      grDevices::dev.off()
    }

    tryCatch({
      save_plot(output_filename, plot_obj, output_type, save_width, save_height)
      message(sprintf("'%s' plot saved successfully to: %s", type, output_filename))
    }, error = function(e) {
      warning(sprintf("Failed to save the plot. Error: %s", e$message))
    })

  } else {
    warning("No plot object was generated, so no file was saved.")
  }

  return(invisible(NULL))
}


# ------------------------------------------------------------------------------
# 4. SHAP Model Explanation Function (figure_shap)
# ------------------------------------------------------------------------------
#' @title Generate and Plot SHAP Explanation Figures
#' @description Trains a surrogate model (XGBoost or Lasso) on the original model's
#'   output scores. It then calculates and visualizes SHAP values to explain each
#'   feature's contribution, providing insights into the original model's behavior.
#'
#' @param data A list object containing model evaluation results. It must contain
#'   `sample_score`, which is a data frame with the sample ID in the first column
#'   and the model's output `score` in another column.
#' @param raw_data A data frame containing the original feature data used for training.
#'   The column structure must be fixed based on `target_type`:
#'   - For "diagnosis": 1st col=ID, 2nd col=Outcome/Label, 3rd+ cols=Features.
#'   - For "prognosis": 1st col=ID, 2nd col=Outcome, 3rd col=Time, 4th+ cols=Features.
#' @param output_file A character string, the base name for the output filename
#'   (without extension).
#' @param model_type A character string specifying the surrogate model for SHAP calculation.
#'   Options: "xgboost" (default) or "lasso".
#' @param output_type A character string, the desired output file format.
#'   Options: "pdf" (default), "png", "svg".
#' @param target_type A character string indicating the analysis type. This determines
#'   how `raw_data` is interpreted. Options: "diagnosis" or "prognosis".
#'
#' @return NULL. The function saves a combined plot (SHAP summary and importance)
#'   to the specified output file.
#' @examples
#' \dontrun{
#' # --- Example for a Diagnosis Model ---
#' # 1. Create dummy raw data and model results
#' set.seed(123)
#' train_dia <- data.frame(
#'   SampleID = paste0("S", 1:100),
#'   Label = sample(c(0, 1), 100, replace = TRUE),
#'   FeatureA = rnorm(100),
#'   FeatureB = runif(100)
#' )
#'
#' bagging_xb_results <- list(
#'   sample_score = data.frame(
#'     ID = paste0("S", 1:100),
#'     score = runif(100, 0, 1) # Dummy scores from a model
#'   )
#' )
#'
#' # 2. Generate SHAP plot
#' # Features used will be 'FeatureA' and 'FeatureB'
#' figure_shap(
#'   data = bagging_xb_results,
#'   raw_data = train_dia,
#'   output_file = "Dia_SHAP_Example",
#'   model_type = "xgboost",
#'   output_type = "pdf",
#'   target_type = "diagnosis"
#' )
#'
#' # --- Example for a Prognosis Model ---
#' # 1. Create dummy raw data and model results
#' train_pro <- data.frame(
#'   PatientID = paste0("P", 1:100),
#'   Status = sample(c(0, 1), 100, replace = TRUE),
#'   Time = runif(100, 50, 2000),
#'   Gene1 = rnorm(100),
#'   ClinicalVar = sample(c("Low", "High"), 100, replace = TRUE)
#' )
#'
#' stacking_gbm_pro_results <- list(
#'   sample_score = data.frame(
#'     ID = paste0("P", 1:100),
#'     score = runif(100, 0, 1) # Dummy scores from a stacking model
#'   )
#' )
#'
#' # 2. Generate SHAP plot
#' # Features used will be 'Gene1' and 'ClinicalVar'
#' figure_shap(
#'   data = stacking_gbm_pro_results,
#'   raw_data = train_pro,
#'   output_file = "Pro_SHAP_Example",
#'   model_type = "lasso",
#'   output_type = "png",
#'   target_type = "prognosis"
#' )
#' }
#' @importFrom dplyr inner_join
#' @importFrom xgboost xgb.DMatrix xgb.train
#' @importFrom glmnet cv.glmnet
#' @importFrom shapviz shapviz sv_importance
#' @importFrom ggplot2 ggplot aes geom_col coord_flip scale_fill_gradient labs
#'   theme_minimal element_text element_blank ggsave
#' @importFrom patchwork plot_layout
#' @importFrom Cairo CairoPDF CairoPNG CairoSVG
#' @importFrom stats reorder complete.cases sd
#' @importFrom utils head
#' @importFrom grDevices dev.off
#' @export
figure_shap <- function(data, raw_data, output_file,
                        model_type = "xgboost", output_type = "pdf",
                        target_type = c("diagnosis", "prognosis")) {

  target_type <- match.arg(target_type)

  # --- 1. Parameter Validation ---
  if (!is.data.frame(raw_data)) {
    stop("'raw_data' must be a data frame.")
  }
  if (!model_type %in% c("xgboost", "lasso")) {
    stop("Invalid 'model_type'. Please choose 'xgboost' or 'lasso'.")
  }
  if (!all(c("sample_score") %in% names(data))) {
    stop("'data' object must contain a 'sample_score' data frame.")
  }
  if (!"score" %in% names(data$sample_score)) {
    stop("'data$sample_score' must contain a 'score' column.")
  }

  # --- 2. Data Preparation ---
  message("Preparing data for SHAP analysis...")
  raw_df <- raw_data
  score_df <- as.data.frame(data$sample_score)

  # Standardize ID columns for merging (always use the first column)
  id_col_raw <- names(raw_df)[1]
  id_col_score <- names(score_df)[1]

  if (id_col_raw != id_col_score) {
    message(paste0("Renaming score data frame ID column from '", id_col_score, "' to '", id_col_raw, "' for merging."))
    names(score_df)[1] <- id_col_raw
  }

  # Merge raw features with model scores
  merged_df <- dplyr::inner_join(raw_df, score_df, by = id_col_raw)

  # Remove rows with NA scores (cannot be used as target for surrogate model)
  if (any(is.na(merged_df$score))) {
    n_removed <- sum(is.na(merged_df$score))
    warning(sprintf("Found and removed %d rows with NA model scores.", n_removed))
    merged_df <- merged_df[!is.na(merged_df$score), ]
  }
  if (nrow(merged_df) == 0) {
    stop("SHAP data frame is empty after removing rows with NA scores.")
  }

  # --- 3. Feature and Target Selection (Corrected Logic) ---
  target_score <- merged_df$score

  # Select feature columns based on position, according to target_type
  if (target_type == "diagnosis") {
    if (ncol(raw_df) < 3) stop("For 'diagnosis', raw_data must have at least 3 columns: ID, Label, and one Feature.")
    feature_cols <- names(raw_df)[-c(1, 2)]
  } else { # target_type == "prognosis"
    if (ncol(raw_df) < 4) stop("For 'prognosis', raw_data must have at least 4 columns: ID, Outcome, Time, and one Feature.")
    feature_cols <- names(raw_df)[-c(1, 2, 3)]
  }

  if (length(feature_cols) == 0) {
    stop("No feature columns were identified. Check your raw_data format and 'target_type'.")
  }

  X_features <- merged_df[, feature_cols, drop = FALSE]
  message(sprintf("Identified %d features for SHAP analysis: %s",
                  length(feature_cols), paste(utils::head(feature_cols, 5), collapse=", ")))

  # --- 4. Data Cleaning for Surrogate Model ---
  # Convert all feature columns to numeric, handling factors/characters
  for (col in names(X_features)) {
    if (!is.numeric(X_features[[col]])) {
      if (is.factor(X_features[[col]])) {
        X_features[[col]] <- as.numeric(X_features[[col]])
      } else {
        X_features[[col]] <- suppressWarnings(as.numeric(as.character(X_features[[col]])))
      }
    }
  }

  # Remove rows with NA values in features (introduced by coercion or originally present)
  if (any(is.na(X_features))) {
    warning("NA values found in feature data. Rows with NAs will be removed. For advanced imputation, please preprocess your data.")
    complete_idx <- stats::complete.cases(X_features)
    n_removed <- sum(!complete_idx)

    X_features <- X_features[complete_idx, , drop = FALSE]
    target_score <- target_score[complete_idx] # Crucially, subset the target as well

    message(sprintf("Removed %d rows from feature data due to NA values.", n_removed))
  }

  if (nrow(X_features) == 0) {
    stop("Feature data is empty after removing NA values. Cannot perform SHAP analysis.")
  }

  X_matrix <- tryCatch({
    data.matrix(X_features)
  }, error = function(e) {
    stop(paste0("Failed to convert feature data to a numeric matrix. Error: ", e$message))
  })

  # --- 5. Train Surrogate Model & Calculate SHAP ---
  message(sprintf("Training '%s' surrogate model and calculating SHAP values...", model_type))

  surrogate_model <- NULL
  if (model_type == "xgboost") {
    dtrain <- xgboost::xgb.DMatrix(X_matrix, label = target_score)
    xgb_params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 3, nthread = 1)
    surrogate_model <- xgboost::xgb.train(params = xgb_params, data = dtrain, nrounds = 100)
  } else if (model_type == "lasso") {
    if (stats::sd(target_score, na.rm = TRUE) < 1e-6) {
      stop("Target score is constant; cannot train a Lasso regression model.")
    }
    surrogate_model <- glmnet::cv.glmnet(X_matrix, target_score, alpha = 1, family = "gaussian")
  }

  if (is.null(surrogate_model)) {
    stop("Surrogate model training failed.")
  }

  sv <- shapviz::shapviz(surrogate_model, X_pred = X_matrix)

  # --- 6. Generate and Save Plots ---
  message("Generating SHAP plots...")

  # SHAP Importance Plot (Bar chart)
  p_bar <- shapviz::sv_importance(sv, kind = "bar", show_numbers = FALSE) +
    ggplot2::labs(
      title = "Feature Importance",
      subtitle = "Mean Absolute SHAP Value",
      x = NULL, y = NULL
    ) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", hjust = 0.5))

  # SHAP Summary Plot (Beeswarm)
  p_beeswarm <- shapviz::sv_importance(sv, kind = "beeswarm", max_display = 15) +
    ggplot2::labs(
      title = "SHAP Summary Plot",
      x = "SHAP value (impact on model score)"
    ) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", hjust = 0.5))

  combined_plot <- p_beeswarm + p_bar + patchwork::plot_layout(ncol = 1, heights = c(2, 1.5))

  # Save the combined plot using Cairo for high-quality, cross-platform output
  output_filename <- paste0(output_file, ".", output_type)
  plot_width <- 10
  plot_height <- 12

  tryCatch({
    if (output_type == "pdf") {
      Cairo::CairoPDF(output_filename, width = plot_width, height = plot_height)
    } else if (output_type == "png") {
      Cairo::CairoPNG(output_filename, width = plot_width * 300, height = plot_height * 300, res = 300)
    } else if (output_type == "svg") {
      Cairo::CairoSVG(output_filename, width = plot_width, height = plot_height)
    } else {
      stop("unsupported") # Fallback to ggsave
    }
    print(combined_plot)
    grDevices::dev.off()
    message(sprintf("SHAP plot successfully saved to: %s", output_filename))
  }, error = function(e) {
    warning(paste0("Cairo-based saving failed. Falling back to ggsave. Error: ", e$message))
    ggplot2::ggsave(output_filename, plot = combined_plot, device = output_type,
                    width = plot_width, height = plot_height, dpi = 300)
    message(sprintf("SHAP plot successfully saved with ggsave to: %s", output_filename))
  })

  return(invisible(NULL))
}
