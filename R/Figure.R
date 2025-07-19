# Figure.R

# Global Aesthetic Color Settings
# These colors are used consistently across plotting functions for branding and clarity.
primary_color <- "#2E86AB"   # Deep blue, often used for primary lines or fills.
secondary_color <- "#A23B72" # Magenta, used for secondary elements or contrasts.
accent_color <- "#F18F01"    # Orange, used for highlighting specific points or annotations.

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

  # Parameter validation
  if (!type %in% c("roc", "prc", "matrix")) {
    stop("Invalid 'type' parameter. Diagnostic model currently supports 'roc', 'prc', or 'matrix'.")
  }
  if (!all(c("sample_score", "evaluation_metrics") %in% names(data))) {
    stop("'data' object format is incorrect; it must contain 'sample_score' and 'evaluation_metrics'.")
  }
  if (!"score" %in% names(data$sample_score)) {
    stop("'data$sample_score' must contain a 'score' column.")
  }

  # Retrieve threshold
  threshold <- data$evaluation_metrics$Final_Threshold
  if (is.null(threshold) || is.na(threshold)) {
    threshold <- data$evaluation_metrics$`_Threshold` # Try alternative threshold name
    if (is.null(threshold) || is.na(threshold)) {
      stop("No valid threshold found. Ensure 'data$evaluation_metrics' contains 'Final_Threshold' or '_Threshold'.")
    }
  }

  df <- as.data.frame(data$sample_score)

  # Ensure 'score' column is numeric and remove NAs
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

  # For ROC and PRC, true labels are required
  if (type %in% c("roc", "prc", "matrix")) { # matrix also needs label
    if (!"label" %in% names(df)) {
      stop(paste0("Plotting ", type, " requires a 'label' column."))
    }

    # Ensure 'label' column is numeric
    tryCatch({
      df$label <- as.numeric(as.character(df$label))
    }, error = function(e) {
      stop(paste0("Failed to convert diagnostic label column to numeric: ", e$message))
    })

    # Remove rows with NA labels
    df <- df[!is.na(df$label), ]
    if (nrow(df) == 0) {
      stop("Diagnostic dataframe is empty after removing NA label values, cannot visualize.")
    }

    # Check if labels are 0 or 1
    unique_labels <- unique(df$label)
    if (!all(unique_labels %in% c(0, 1)) || length(unique_labels) < 2) {
      stop("Diagnostic labels (label) must exclusively contain 0 and 1, and both must be present for meaningful plots.")
    }
  }

  output_filename <- paste0(output_file, ".", output_type)
  plot_obj <- NULL

  if (type == "roc") {
    # Plot ROC curve
    roc_obj <- pROC::roc(df$label, df$score, quiet = TRUE)
    auc_value <- as.numeric(roc_obj$auc)

    roc_data <- data.frame(
      specificity = roc_obj$specificities,
      sensitivity = roc_obj$sensitivities
    )

    # Calculate sensitivity and specificity at the given threshold
    coords_at_threshold <- pROC::coords(roc_obj, x = threshold, input = "threshold",
                                        ret = c("sensitivity", "specificity"))

    plot_obj <- ggplot2::ggplot(roc_data, ggplot2::aes(x = 1 - specificity, y = sensitivity)) +
      ggplot2::geom_line(color = primary_color, linewidth = 1.2) +
      ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
      # Mark threshold point
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
      )

  } else if (type == "prc") {
    # Plot PRC curve
    prc_obj <- PRROC::pr.curve(scores.class0 = df$score[df$label == 1],
                               scores.class1 = df$score[df$label == 0],
                               curve = TRUE)
    auprc_value <- prc_obj$auc.integral

    prc_data <- data.frame(
      recall = prc_obj$curve[, 1],
      precision = prc_obj$curve[, 2]
    )

    # Calculate precision and recall at the given threshold
    predicted_labels <- base::ifelse(df$score > threshold, 1, 0)
    tp <- sum(predicted_labels == 1 & df$label == 1)
    fp <- sum(predicted_labels == 1 & df$label == 0)
    fn <- sum(predicted_labels == 0 & df$label == 1)

    precision_at_threshold <- base::ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall_at_threshold <- base::ifelse(tp + fn > 0, tp / (tp + fn), 0)

    plot_obj <- ggplot2::ggplot(prc_data, ggplot2::aes(x = recall, y = precision)) +
      ggplot2::geom_line(color = secondary_color, linewidth = 1.2) +
      # Mark threshold point
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
      )

  } else if (type == "matrix") {
    # Plot Confusion Matrix
    # Predict labels based on threshold
    predicted_labels <- base::ifelse(df$score > threshold, 1, 0)

    # Calculate confusion matrix
    cm_table <- table(Predicted = factor(predicted_labels, levels = c(0, 1)),
                      Actual = factor(df$label, levels = c(0, 1)))

    # Convert to dataframe for ggplot
    cm_df <- as.data.frame(cm_table)

    # Calculate percentages
    cm_df$Percentage <- cm_df$Freq / sum(cm_df$Freq) * 100

    # Calculate performance metrics
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

  # Save plot using Cairo for high-quality output
  if (!is.null(plot_obj)) {
    if (output_type == "pdf") {
      Cairo::CairoPDF(output_filename, width = 9, height = 7)
    } else if (output_type == "png") {
      Cairo::CairoPNG(output_filename, width = 9 * 300, height = 7 * 300, res = 300)
    } else if (output_type == "svg") {
      Cairo::CairoSVG(output_filename, width = 9, height = 7)
    } else {
      warning(paste0("Unsupported output type: ", output_type, ". Attempting ggsave fallback."))
      ggplot2::ggsave(output_filename, plot = plot_obj, device = output_type,
                      width = 9, height = 7, dpi = 300)
      message(sprintf("'%s' plot saved to: %s", type, output_filename))
      return(invisible(NULL))
    }
    print(plot_obj) # Print to Cairo device
    grDevices::dev.off()
    message(sprintf("'%s' plot saved to: %s", type, output_filename))
  } else {
    warning("No plot object generated, file not saved.")
  }
}


# ------------------------------------------------------------------------------
# 3. Prognostic Model Visualization Function (figure_pro)
# ------------------------------------------------------------------------------
#' @title Plot Prognostic Model Evaluation Figures
#' @description Generates Kaplan-Meier (KM) survival curves or time-dependent ROC curves
#'   based on prognostic model evaluation results.
#'
#' @param type String, specifies the type of plot to generate. Options are "km"
#'   (Kaplan-Meier curve) or "tdroc" (Time-Dependent ROC curve).
#' @param data A list object containing model evaluation results. It must include:
#'   \itemize{
#'     \item `sample_score`: A data frame with at least "ID", "time", "outcome" (0/1), and "score" columns.
#'     \item `evaluation_metrics`: A list with a "KM_Cutoff" value for KM plots, and
#'       "AUROC_Years" (a numeric vector of evaluation years) for tdROC plots.
#'   }
#' @param output_file String, the base name for the output file (without extension).
#' @param output_type String, the desired output file format. Options: "pdf", "png", "svg".
#'   Defaults to "pdf".
#' @param time_unit String, specifies the unit of time for the x-axis label in tdROC plots
#'   and for internal time conversions (e.g., "days", "months", "years"). Defaults to "days".
#'
#' @return NULL. The function saves the generated plot directly to a file.
#' @examples
#' \dontrun{
#' # Example data structure for prognostic model evaluation results:
#' external_eval_example_pro <- list(
#'   sample_score = data.frame(
#'     ID = paste0("S", 1:100),
#'     time = runif(100, 10, 1000), # time in days
#'     outcome = sample(c(0, 1), 100, replace = TRUE),
#'     score = runif(100, 0, 1)
#'   ),
#'   evaluation_metrics = list(
#'     KM_Cutoff = 0.5,
#'     AUROC_Years = c(1, 2, 3, 5) # Evaluation years
#'   )
#' )
#'
#' # Plot Kaplan-Meier curve
#' figure_pro(type = "km", data = external_eval_example_pro,
#'            output_file = "Prognostic_Model_KM", output_type = "svg",
#'            time_unit = "days")
#'
#' # Plot Time-Dependent ROC curve
#' figure_pro(type = "tdroc", data = external_eval_example_pro,
#'            output_file = "Prognostic_Model_TDROC", output_type = "pdf",
#'            time_unit = "days") # Assume sample_score$time is in days, AUROC_Years is in years
#' }
#' @importFrom survival Surv survfit
#' @importFrom survminer ggsurvplot
#' @importFrom survivalROC survivalROC
#' @importFrom ggplot2 ggplot aes geom_line geom_point geom_text labs
#'   scale_y_continuous theme_bw element_text element_blank ggsave
#' @importFrom Cairo CairoPDF CairoPNG CairoSVG
#' @export
figure_pro <- function(type, data, output_file, output_type = "pdf", time_unit = "days") {

  # Parameter validation
  if (!type %in% c("km", "tdroc")) {
    stop("Invalid 'type' parameter. Prognostic model currently supports 'km' or 'tdroc'.")
  }
  if (!all(c("sample_score", "evaluation_metrics") %in% names(data))) {
    stop("'data' object format is incorrect; it must contain 'sample_score' and 'evaluation_metrics'.")
  }
  if (!all(c("time", "outcome", "score") %in% names(data$sample_score))) {
    stop("'data$sample_score' must contain 'time', 'outcome', and 'score' columns.")
  }

  df <- as.data.frame(data$sample_score)

  # Ensure time, outcome, score columns are numeric and remove NAs
  initial_rows <- nrow(df)
  tryCatch({
    df$time <- as.numeric(as.character(df$time))
    df$outcome <- as.numeric(as.character(df$outcome))
    df$score <- as.numeric(as.character(df$score))
  }, error = function(e) {
    stop(paste0("Failed to convert prognostic model columns to numeric: ensure 'time', 'outcome', and 'score' are convertible to numeric. Error: ", e$message))
  })

  df <- df[!is.na(df$time) & !is.na(df$outcome) & !is.na(df$score), ]
  if (nrow(df) == 0) {
    stop("Prognostic dataframe is empty after removing NA values, cannot visualize.")
  }
  if (nrow(df) < initial_rows) {
    warning(sprintf("Removed %d rows from prognostic data due to NA values for visualization.", initial_rows - nrow(df)))
  }

  # Strictly check and filter 'outcome' column to ensure only 0 and 1
  invalid_outcome_rows <- !df$outcome %in% c(0, 1)
  if (any(invalid_outcome_rows)) {
    num_invalid_rows <- sum(invalid_outcome_rows)
    warning(sprintf("Found %d rows in prognostic outcome column with values other than 0 or 1; these rows will be removed for correct plotting.",
                    num_invalid_rows))
    df <- df[!invalid_outcome_rows, ]
  }

  # Check if outcomes are 0 or 1 and both are present
  unique_outcomes <- unique(df$outcome)
  if (!all(unique_outcomes %in% c(0, 1)) || length(unique_outcomes) < 2) {
    stop("Prognostic outcomes (outcome) must exclusively contain 0 and 1, and both must be present for meaningful plots.")
  }

  output_filename <- paste0(output_file, ".", output_type)
  plot_obj <- NULL

  if (type == "km") {
    cutoff <- data$evaluation_metrics$KM_Cutoff
    if (is.null(cutoff) || is.na(cutoff)) {
      stop("Cannot plot KM curve; 'KM_Cutoff' is invalid in data$evaluation_metrics.")
    }

    df$risk_group <- base::ifelse(df$score > cutoff, "High Risk", "Low Risk")
    df$risk_group <- factor(df$risk_group, levels = c("Low Risk", "High Risk"))

    fit <- survival::survfit(survival::Surv(time, outcome) ~ risk_group, data = df)

    # Plot KM curve
    km_plot <- survminer::ggsurvplot(
      fit,
      data = df,
      pval = TRUE,
      conf.int = TRUE,
      risk.table = TRUE,
      risk.table.col = "strata",
      risk.table.y.text = FALSE,
      xlab = paste0("Time (", time_unit, ")"),
      ylab = "Overall Survival Probability",
      title = "Kaplan-Meier Survival Curve",
      legend.title = "Risk Group",
      legend.labs = c("Low Risk", "High Risk"),
      palette = c("#0073C2FF", "#E7B800FF"),
      ggtheme = ggplot2::theme_bw() + ggplot2::theme(
        plot.title = ggplot2::element_text(size = 18, face = "bold", hjust = 0.5),
        axis.title = ggplot2::element_text(size = 14, face = "bold"),
        axis.text = ggplot2::element_text(size = 12),
        legend.title = ggplot2::element_text(size = 12, face = "bold"),
        legend.text = ggplot2::element_text(size = 11)
      )
    )
    plot_obj <- km_plot # ggsurvplot returns a list, requires special handling for saving
  } else if (type == "tdroc") {
    # Time-dependent ROC curve (plot specific years based on AUROC_Years)
    if (!"AUROC_Years" %in% names(data$evaluation_metrics) ||
        !is.numeric(data$evaluation_metrics$AUROC_Years) ||
        length(data$evaluation_metrics$AUROC_Years) == 0) {
      stop("Plotting time-dependent ROC requires 'AUROC_Years' to provide valid numeric evaluation time points in data$evaluation_metrics.")
    }

    # Set conversion factor based on time_unit parameter
    # AUROC_Years are assumed to be in years. `time_unit` refers to the unit of `df$time`.
    # Convert AUROC_Years to match `df$time`'s unit for `survivalROC`.
    time_conversion_factor_to_df_unit <- 1
    display_time_unit_label <- "years" # Default display for x-axis will be years (source of AUROC_Years)

    if (time_unit == "days") {
      time_conversion_factor_to_df_unit <- 365.25 # 1 year = 365.25 days
      display_time_unit_label <- "years" # Still display x-axis in years
    } else if (time_unit == "months") {
      time_conversion_factor_to_df_unit <- 12 # 1 year = 12 months
      display_time_unit_label <- "years" # Still display x-axis in years
    } else if (time_unit == "years") {
      time_conversion_factor_to_df_unit <- 1 # df$time is already in years
      display_time_unit_label <- "years"
    } else {
      warning("Unrecognized 'time_unit' for prognostic model. Assuming 'years' for AUROC_Years conversion.")
      time_conversion_factor_to_df_unit <- 1
      display_time_unit_label <- "years"
    }

    eval_time_points <- sort(unique(data$evaluation_metrics$AUROC_Years)) # These are in years
    eval_time_points_converted <- eval_time_points * time_conversion_factor_to_df_unit # Converted to df$time's unit


    roc_data_list <- list()
    max_observed_time <- max(df$time, na.rm = TRUE)
    min_event_time <- min(df$time[df$outcome == 1], na.rm = TRUE)

    if (is.infinite(min_event_time) || is.infinite(max_observed_time) || min_event_time > max_observed_time) {
      stop("Invalid data time range or event data; cannot calculate time-dependent ROC curve.")
    }

    for (i in seq_along(eval_time_points_converted)) {
      t_point_converted <- eval_time_points_converted[i]
      t_point_original_year <- eval_time_points[i] # For display

      # Check if evaluation time point is reasonable
      if (t_point_converted < min_event_time) {
        warning(sprintf("Evaluation time point %.2f (approx. %.1f %s) is earlier than the minimum observed event time %.2f %s; skipping this time point.",
                        t_point_original_year, t_point_converted, time_unit, min_event_time, time_unit))
        next
      }
      if (t_point_converted > max_observed_time) {
        warning(sprintf("Evaluation time point %.2f (approx. %.1f %s) is later than the maximum observed time %.2f %s, potentially leading to unstable calculations, but will attempt.",
                        t_point_original_year, t_point_converted, time_unit, max_observed_time, time_unit))
      }

      roc_result <- tryCatch({
        survivalROC::survivalROC(
          Stime = df$time,
          status = df$outcome,
          marker = df$score,
          predict.time = t_point_converted, # Use the converted time point here
          method = "KM" # Or "NNE"
        )
      }, error = function(e) {
        warning(paste("Failed to calculate ROC for time point", round(t_point_original_year, 2), "years:", e$message))
        return(NULL)
      })

      if (!is.null(roc_result) && !is.na(roc_result$AUC)) {
        roc_data_list[[as.character(t_point_original_year)]] <- data.frame(
          time = t_point_original_year, # Plot X-axis shows original year unit
          AUROC = roc_result$AUC,
          label = paste0(round(t_point_original_year, 1), " ",
                         display_time_unit_label,
                         " (AUROC: ", sprintf("%.3f", roc_result$AUC), ")")
        )
      }
    }

    if (length(roc_data_list) == 0) {
      stop("Failed to calculate AUROC for any valid time points; cannot plot time-dependent ROC curve. Check if AUROC_Years specifies time points within data range and with events.")
    }

    tdroc_df <- do.call(rbind, roc_data_list)

    # Plot time-dependent ROC curve
    plot_obj <- ggplot2::ggplot(tdroc_df, ggplot2::aes(x = time, y = AUROC)) +
      ggplot2::geom_line(color = "darkgreen", linewidth = 1.2) +
      ggplot2::geom_point(color = "darkgreen", size = 3) +
      ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", AUROC)), hjust = -0.2, vjust = 0.5, size = 3, color = "darkgreen") + # Display AUROC value
      ggplot2::scale_y_continuous(limits = c(0.5, 1), breaks = seq(0.5, 1, 0.1)) +
      ggplot2::labs(
        title = "Time-Dependent AUROC Curve at Specific Time Points",
        x = paste0("Time (", display_time_unit_label, ")"), # X-axis label indicates original year unit
        y = "Area Under ROC Curve (AUROC)"
      ) +
      ggplot2::theme_bw() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(size = 18, face = "bold", hjust = 0.5),
        axis.title = ggplot2::element_text(size = 14, face = "bold"),
        axis.text = ggplot2::element_text(size = 12),
        panel.grid.minor = ggplot2::element_blank()
      )
  }

  # Save plot
  if (!is.null(plot_obj)) {
    if (type == "km") {
      # ggsurvplot returns a list, requiring special handling
      if (output_type == "pdf") {
        Cairo::CairoPDF(output_filename, width = 9, height = 8)
      } else if (output_type == "png") {
        Cairo::CairoPNG(output_filename, width = 9 * 300, height = 8 * 300, res = 300)
      } else if (output_type == "svg") {
        Cairo::CairoSVG(output_filename, width = 9, height = 8)
      } else {
        warning(paste0("Unsupported output type for KM plot: ", output_type, ". Attempting ggsave fallback."))
        ggplot2::ggsave(output_filename, plot = plot_obj, device = output_type,
                        width = 9, height = 8, dpi = 300)
        message(sprintf("'%s' plot saved to: %s", type, output_filename))
        return(invisible(NULL))
      }
      print(plot_obj, newpage = FALSE) # Print to Cairo device
      grDevices::dev.off()
    } else {
      # Use Cairo for high-quality output
      if (output_type == "pdf") {
        Cairo::CairoPDF(output_filename, width = 9, height = 7)
      } else if (output_type == "png") {
        Cairo::CairoPNG(output_filename, width = 9 * 300, height = 7 * 300, res = 300)
      } else if (output_type == "svg") {
        Cairo::CairoSVG(output_filename, width = 9, height = 7)
      } else {
        warning(paste0("Unsupported output type: ", output_type, ". Attempting ggsave fallback."))
        ggplot2::ggsave(output_filename, plot = plot_obj, device = output_type,
                        width = 9, height = 7, dpi = 300)
        message(sprintf("'%s' plot saved to: %s", type, output_filename))
        return(invisible(NULL))
      }
      print(plot_obj) # Print to Cairo device
      grDevices::dev.off()
    }
    message(sprintf("'%s' plot saved to: %s", type, output_filename))
  } else {
    warning("No plot object generated, file not saved.")
  }
}


# ------------------------------------------------------------------------------
# 4. SHAP Model Explanation Function (figure_shap)
# ------------------------------------------------------------------------------
#' @title Generate and Plot SHAP Explanation Figures
#' @description Trains a surrogate model (XGBoost or Lasso) to approximate the original
#'              model's output scores. It then calculates and visualizes SHAP (SHapley Additive
#'              exPlanations) values to explain each feature's contribution to the
#'              surrogate model's predictions, thereby providing insights into the
#'              original model's behavior.
#'
#' @param data A list object containing model evaluation results. It must contain
#'   `sample_score`, which is a data frame with at least "sample" (or "ID") and "score" columns.
#' @param raw_data_path String, the file path to the original feature data in CSV format.
#'   The first column of this CSV should be the sample ID (matching "sample" or "ID" in `data$sample_score`).
#' @param output_file String, the base name for the output filename (without extension).
#' @param model_type String, specifies the type of surrogate model to train for SHAP
#'   value calculation. Options: "xgboost" (default) or "lasso".
#' @param output_type String, the desired output file format. Options: "pdf", "png", "svg".
#'   Defaults to "pdf".
#' @param target_type String, indicates whether the SHAP analysis is for a
#'   "diagnosis" or "prognosis" model. This helps in correctly identifying and
#'   excluding non-feature columns (like "label", "outcome", "time") from the raw data.
#'
#' @return NULL. The function saves a combined plot (SHAP beeswarm and importance bar chart)
#'   to the specified output file.
#' @examples
#' \dontrun{
#' # Create dummy raw data and model evaluation results for demonstration
#' # For diagnostic model:
#' set.seed(123)
#' dummy_raw_dia_data <- data.frame(
#'   sample = paste0("S", 1:100),
#'   featureA = rnorm(100),
#'   featureB = runif(100),
#'   featureC = sample(1:10, 100, replace = TRUE),
#'   label = sample(c(0, 1), 100, replace = TRUE) # Example label column
#' )
#' write.csv(dummy_raw_dia_data, "dummy_raw_dia_data.csv", row.names = FALSE)
#'
#' dummy_eval_dia_results <- list(
#'   sample_score = data.frame(
#'     sample = paste0("S", 1:100),
#'     score = runif(100, 0, 1) # Example model scores
#'   )
#' )
#'
#' # Run SHAP for a diagnostic model
#' figure_shap(data = dummy_eval_dia_results,
#'             raw_data_path = "dummy_raw_dia_data.csv",
#'             output_file = "Diagnostic_SHAP_XGBoost",
#'             model_type = "xgboost",
#'             output_type = "pdf",
#'             target_type = "diagnosis")
#'
#' # For prognostic model:
#' dummy_raw_pro_data <- data.frame(
#'   ID = paste0("P", 1:100),
#'   gene1 = rnorm(100),
#'   gene2 = runif(100),
#'   clinic_var = sample(c(10, 20), 100, replace = TRUE),
#'   time = runif(100, 50, 1000), # Example time column
#'   outcome = sample(c(0, 1), 100, replace = TRUE) # Example outcome column
#' )
#' write.csv(dummy_raw_pro_data, "dummy_raw_pro_data.csv", row.names = FALSE)
#'
#' dummy_eval_pro_results <- list(
#'   sample_score = data.frame(
#'     ID = paste0("P", 1:100),
#'     score = runif(100, 0, 1) # Example model scores
#'   )
#' )
#'
#' # Run SHAP for a prognostic model
#' figure_shap(data = dummy_eval_pro_results,
#'             raw_data_path = "dummy_raw_pro_data.csv",
#'             output_file = "Prognostic_SHAP_Lasso",
#'             model_type = "lasso",
#'             output_type = "png",
#'             target_type = "prognosis")
#'
#' # Clean up dummy files
#' unlink("dummy_raw_dia_data.csv")
#' unlink("dummy_raw_pro_data.csv")
#' }
#' @importFrom readr read_csv
#' @importFrom dplyr inner_join
#' @importFrom xgboost xgb.DMatrix xgb.train
#' @importFrom glmnet cv.glmnet
#' @importFrom shapviz shapviz sv_importance
#' @importFrom ggplot2 ggplot aes geom_col coord_flip scale_fill_gradient labs
#'   theme_minimal element_text element_blank
#' @importFrom patchwork plot_layout
#' @importFrom Cairo CairoPDF CairoPNG CairoSVG
#' @export
figure_shap <- function(data, raw_data_path, output_file,
                        model_type = "xgboost", output_type = "pdf",
                        target_type = c("diagnosis", "prognosis")) {

  target_type <- match.arg(target_type)

  # Parameter validation
  if (!model_type %in% c("xgboost", "lasso")) {
    stop("Invalid 'model_type' parameter. Please choose 'xgboost' or 'lasso'.")
  }
  if (!file.exists(raw_data_path)) {
    stop(paste("Raw data file not found:", raw_data_path))
  }
  if (!all(c("sample_score") %in% names(data))) {
    stop("'data' object format is incorrect; it must contain 'sample_score'.")
  }
  if (!"score" %in% names(data$sample_score)) {
    stop("'data$sample_score' must contain a 'score' column.")
  }

  # Prepare data
  message("Preparing data for SHAP analysis...")
  raw_df <- readr::read_csv(raw_data_path, show_col_types = FALSE)
  score_df <- as.data.frame(data$sample_score)

  # Standardize ID column names for merging
  # Attempt to find common ID names, otherwise default to first column
  common_id_names <- c("sample", "ID")
  id_col_raw <- intersect(names(raw_df), common_id_names)
  id_col_score <- intersect(names(score_df), common_id_names)

  if (length(id_col_raw) == 0) {
    id_col_raw <- names(raw_df)[1] # Default to first column
    warning(paste0("No explicit ID column ('sample' or 'ID') recognized in raw data; defaulting to first column '", id_col_raw, "' as ID."))
  } else {
    id_col_raw <- id_col_raw[1] # Take the first match if multiple exist
  }
  if (length(id_col_score) == 0) {
    id_col_score <- names(score_df)[1] # Default to first column
    warning(paste0("No explicit ID column ('sample' or 'ID') recognized in model scores; defaulting to first column '", id_col_score, "' as ID."))
  } else {
    id_col_score <- id_col_score[1] # Take the first match if multiple exist
  }

  # Ensure ID column names are identical for merging
  if (id_col_raw != id_col_score) {
    # If different, rename one to match the other, prioritizing the raw_df's ID name
    names(score_df)[names(score_df) == id_col_score] <- id_col_raw
    message(paste0("Renamed score dataframe ID column from '", id_col_score, "' to '", id_col_raw, "' for merging."))
  }


  # Merge data
  merged_df <- dplyr::inner_join(raw_df, score_df, by = id_col_raw)

  # Clean NA values in target score
  initial_rows_shap <- nrow(merged_df)
  if (any(is.na(merged_df$score))) {
    warning("NA values found in model scores for SHAP analysis; these rows will be removed.")
    merged_df <- merged_df[!is.na(merged_df$score), ]
  }
  if (nrow(merged_df) == 0) {
    stop("SHAP analysis dataframe is empty after removing NA values in scores.")
  }
  if (nrow(merged_df) < initial_rows_shap) {
    message(sprintf("Removed %d rows from merged data due to NA values in scores.", initial_rows_shap - nrow(merged_df)))
  }

  # Prepare surrogate model input
  target_score <- merged_df$score

  # Identify feature columns (excluding ID, label/outcome/time, and score)
  exclude_cols <- c(id_col_raw, "score")
  if (target_type == "diagnosis") {
    exclude_cols <- c(exclude_cols, "label")
  } else if (target_type == "prognosis") {
    exclude_cols <- c(exclude_cols, "outcome", "time")
  }

  feature_cols <- setdiff(names(merged_df), exclude_cols)
  X_features <- merged_df[, feature_cols, drop = FALSE]

  # Ensure all feature columns are numeric and handle NAs
  initial_X_rows <- nrow(X_features)
  # Check for non-numeric columns and attempt conversion
  for (col in names(X_features)) {
    if (!is.numeric(X_features[[col]])) {
      X_features[[col]] <- suppressWarnings(as.numeric(as.character(X_features[[col]])))
      if (any(is.na(X_features[[col]]))) {
        warning(paste0("Column '", col, "' converted to numeric but introduced NAs. These NAs will be handled later."))
      }
    }
  }

  if (any(is.na(X_features))) {
    warning("NA values found in feature data for SHAP analysis; rows containing NAs will be removed. For more complex missing value imputation, preprocess raw data before calling this function.")
    complete_cases_idx <- stats::complete.cases(X_features)
    X_features <- X_features[complete_cases_idx, ]
    target_score <- target_score[complete_cases_idx] # Adjust target_score accordingly
  }

  if (nrow(X_features) == 0) {
    stop("Feature dataframe is empty after removing NA values, cannot perform SHAP analysis.")
  }
  if (nrow(X_features) < initial_X_rows) {
    message(sprintf("Removed %d rows from feature data due to NA values.", initial_X_rows - nrow(X_features)))
  }


  X_matrix <- tryCatch({
    data.matrix(X_features)
  }, error = function(e) {
    stop(paste0("Failed to convert feature data to numeric matrix. Check if feature columns contain non-numeric data that cannot be coerced: ", e$message))
  })

  # Train surrogate model and calculate SHAP values
  message(sprintf("Training '%s' surrogate model and calculating SHAP values...", model_type))

  surrogate_model <- NULL
  if (model_type == "xgboost") {
    dtrain <- xgboost::xgb.DMatrix(X_matrix, label = target_score)
    xgb_params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 3, nthread = 1) # nthread=1 for reproducibility/simplicity
    surrogate_model <- xgboost::xgb.train(params = xgb_params, data = dtrain, nrounds = 100)

  } else if (model_type == "lasso") {
    # For Lasso, ensure target_score is not constant, otherwise cv.glmnet will error
    if (stats::sd(target_score, na.rm = TRUE) == 0) {
      stop("Target score (model score) is constant; cannot train Lasso regression model. Please check your model output.")
    }
    surrogate_model <- glmnet::cv.glmnet(X_matrix, target_score, alpha = 1, family = "gaussian")
  }

  if (is.null(surrogate_model)) {
    stop("Surrogate model training failed.")
  }

  # Calculate SHAP values using shapviz
  sv <- shapviz::shapviz(surrogate_model, X_pred = X_matrix)

  # Plot figures
  message("Generating SHAP plots...")

  # 1. SHAP Importance Plot (Bar chart)
  # Use sv_importance to get the underlying data for custom ggplot
  p_importance_raw <- shapviz::sv_importance(sv, kind = "bar", show_numbers = TRUE)

  # Extract data from the internal ggplot object for full customization
  importance_data <- p_importance_raw$data
  p_bar <- ggplot2::ggplot(importance_data, ggplot2::aes(x = stats::reorder(feature, value), y = value, fill = value)) +
    ggplot2::geom_col(show.legend = FALSE) +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_gradient(low = "lightblue", high = primary_color) + # Use primary color for gradients
    ggplot2::labs(
      title = "Feature Importance",
      subtitle = "Mean(|SHAP value|)",
      x = "Feature",
      y = "Mean Absolute SHAP Value"
    ) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5),
      plot.subtitle = ggplot2::element_text(hjust = 0.5),
      panel.grid.minor = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
    )

  # 2. SHAP Beeswarm Plot
  p_beeswarm <- shapviz::sv_importance(sv, kind = "beeswarm", max_display = 15) +
    ggplot2::labs(
      title = "SHAP Summary Plot",
      x = "SHAP value (impact on model output score)"
    ) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5),
      legend.position = "right"
    )

  # Combine and save plots
  combined_plot <- p_beeswarm + p_bar + patchwork::plot_layout(ncol = 1, heights = c(2, 1))

  output_filename <- paste0(output_file, ".", output_type)

  # Save plot using Cairo for high-quality output
  if (!is.null(combined_plot)) {
    if (output_type == "pdf") {
      Cairo::CairoPDF(output_filename, width = 10, height = 12)
    } else if (output_type == "png") {
      Cairo::CairoPNG(output_filename, width = 10 * 300, height = 12 * 300, res = 300)
    } else if (output_type == "svg") {
      Cairo::CairoSVG(output_filename, width = 10, height = 12)
    } else {
      warning(paste0("Unsupported output type: ", output_type, ". Attempting ggsave fallback."))
      ggplot2::ggsave(output_filename, plot = combined_plot, device = output_type,
                      width = 10, height = 12, dpi = 300)
      message(sprintf("SHAP combined plot saved to: %s", output_filename))
      return(invisible(NULL))
    }
    print(combined_plot) # Print to Cairo device
    grDevices::dev.off()
    message(sprintf("SHAP combined plot saved to: %s", output_filename))
  } else {
    warning("No combined SHAP plot object generated, file not saved.")
  }
}

