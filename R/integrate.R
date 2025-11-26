# ==============================================================================
# integrate.R (2025.11.23-update)
# ------------------------------------------------------------------------------

#' @importFrom stats setNames
#' @importFrom dplyr all_of
#' @importFrom ggplot2 scale_fill_manual
#' @importFrom utils globalVariables
utils::globalVariables(c("AUROC", "Category",'Model','Value',"Dataset", "Average"))

# ==============================================================================
# Diagnostic Model Integration
# ==============================================================================

#' @title Comprehensive Diagnostic Modeling Pipeline
#' @description Executes a complete diagnostic modeling workflow including single
#'   models, bagging, stacking, and voting ensembles across training and multiple
#'   test datasets. Returns structured results with AUROC values for visualization.
#'
#' @param ... Data frames for analysis. The first is the training dataset; all
#'   subsequent arguments are test datasets.
#' @param model_names Character vector specifying which models to use.
#'   If NULL (default), uses all registered models.
#' @param tune Logical, enable hyperparameter tuning. Default TRUE.
#' @param n_estimators Integer, number of bootstrap samples for bagging. Default 10.
#' @param seed Integer for reproducibility. Default 123.
#' @param positive_label_value Value representing positive class. Default 1.
#' @param negative_label_value Value representing negative class. Default 0.
#' @param new_positive_label Factor level name for positive class. Default "Positive".
#' @param new_negative_label Factor level name for negative class. Default "Negative".
#'
#' @return A list containing all_results, auroc_matrix, model_categories, dataset_names.
#'
#' @export
int_dia <- function(...,
                    model_names = NULL,
                    tune = TRUE,
                    n_estimators = 10,
                    seed = 123,
                    positive_label_value = 1,
                    negative_label_value = 0,
                    new_positive_label = "Positive",
                    new_negative_label = "Negative") {

  datasets <- list(...)
  n_datasets <- length(datasets)
  if (n_datasets < 1) stop("At least one dataset (training) must be provided.")

  dataset_names <- c("Train", if (n_datasets > 1) paste0("Test", seq_len(n_datasets - 1)) else NULL)
  train_data <- datasets[[1]]

  initialize_modeling_system_dia()

  if (is.null(model_names)) {
    all_base_models <- names(get_registered_models_dia())
  } else {
    all_base_models <- model_names
    registered <- names(get_registered_models_dia())
    invalid_models <- setdiff(all_base_models, registered)
    if (length(invalid_models) > 0) {
      stop(sprintf("The following models are not registered: %s\nRegistered models are: %s",
                   paste(invalid_models, collapse = ", "),
                   paste(registered, collapse = ", ")))
    }
  }

  all_trained_models <- list()
  model_categories <- c()

  set.seed(seed)
  single_results <- tryCatch({
    models_dia(
      data = train_data, model = all_base_models, tune = tune, seed = seed,
      threshold_choices = "youden", positive_label_value = positive_label_value,
      negative_label_value = negative_label_value, new_positive_label = new_positive_label,
      new_negative_label = new_negative_label
    )
  }, error = function(e) {
    warning(sprintf("Single models training failed: %s. Skipping all single models.", e$message))
    list()
  })

  if (length(single_results) > 0) {
    all_trained_models <- c(all_trained_models, single_results)
    model_categories <- c(model_categories, setNames(rep("Single", length(single_results)), names(single_results)))
  } else {
    warning("No single models were successfully trained. Subsequent ensemble models may fail.")
  }

  # Bagging
  for (bm in all_base_models) {
    bag_name <- paste0("Bagging_", bm)

    bag_result <- tryCatch({
      bagging_dia(
        data = train_data, base_model_name = bm, n_estimators = n_estimators,
        subset_fraction = 0.632, tune_base_model = tune, threshold_choices = "f1",
        positive_label_value = positive_label_value, negative_label_value = negative_label_value,
        new_positive_label = new_positive_label, new_negative_label = new_negative_label, seed = seed
      )
    }, error = function(e) {
      warning(sprintf("Bagging model '%s' failed: %s. Skipping this model.", bag_name, e$message))
      NULL
    })

    if (!is.null(bag_result)) {
      all_trained_models[[bag_name]] <- bag_result
      model_categories[bag_name] <- "Bagging"
    }
  }

  # Stacking
  # Stacking
  if (length(single_results) < 3) {
    warning("Not enough single models for stacking (need at least 3). Skipping all stacking models.")
  } else {
    aurocs <- sapply(single_results, function(r) r$evaluation_metrics$AUROC)
    top10 <- names(sort(aurocs, decreasing = TRUE))[1:min(10, length(aurocs))]
    top5 <- names(sort(aurocs, decreasing = TRUE))[1:min(5, length(aurocs))]

    total_stacking <- length(all_base_models) * 2
    current_stacking <- 0

    for (top_set in list(list(name = "10", models = top10), list(name = "5", models = top5))) {
      for (meta_model in all_base_models) {
        current_stacking <- current_stacking + 1
        stack_name <- paste0("Stacking(", top_set$name, ")_", meta_model)


        message(sprintf("Training stacking model %d/%d: %s",
                        current_stacking, total_stacking, stack_name))

        stack_result <- tryCatch({
          stacking_dia(
            results_all_models = single_results[top_set$models],
            data = train_data,
            meta_model_name = meta_model,
            top = length(top_set$models),
            tune_meta = tune,
            threshold_choices = "youden",
            seed = seed,
            positive_label_value = positive_label_value,
            negative_label_value = negative_label_value,
            new_positive_label = new_positive_label,
            new_negative_label = new_negative_label
          )
        }, error = function(e) {
          warning(sprintf("Stacking model '%s' failed: %s. Skipping this model.", stack_name, e$message))
          NULL
        })

        if (!is.null(stack_result)) {
          all_trained_models[[stack_name]] <- stack_result
          model_categories[stack_name] <- paste0("Stacking(", top_set$name, ")")
        }
      }
    }
  }

  # Voting
  if (length(single_results) < 3) {
    warning("Not enough single models for voting (need at least 3). Skipping all voting models.")
  } else {
    aurocs <- sapply(single_results, function(r) r$evaluation_metrics$AUROC)
    top10 <- names(sort(aurocs, decreasing = TRUE))[1:min(10, length(aurocs))]
    top5 <- names(sort(aurocs, decreasing = TRUE))[1:min(5, length(aurocs))]

    for (top_set in list(list(name = "10", models = top10), list(name = "5", models = top5))) {
      for (vote_type in c("soft", "hard")) {
        vote_name <- paste0("Voting(", top_set$name, ")_", vote_type)

        vote_result <- tryCatch({
          voting_dia(
            results_all_models = single_results[top_set$models], data = train_data,
            type = vote_type, weight_metric = "AUROC", top = length(top_set$models),
            threshold_choices = "youden", seed = seed,
            positive_label_value = positive_label_value,
            negative_label_value = negative_label_value,
            new_positive_label = new_positive_label,
            new_negative_label = new_negative_label
          )
        }, error = function(e) {
          warning(sprintf("Voting model '%s' failed: %s. Skipping this model.", vote_name, e$message))
          NULL
        })

        if (!is.null(vote_result)) {
          all_trained_models[[vote_name]] <- vote_result
          model_categories[vote_name] <- paste0("Voting(", top_set$name, ")")
        }
      }
    }
  }

  auroc_matrix <- matrix(NA, nrow = length(all_trained_models), ncol = n_datasets,
                         dimnames = list(names(all_trained_models), dataset_names))

  for (i in seq_along(datasets)) {
    for (mn in names(all_trained_models)) {

      if (i == 1) {
        evals <- all_trained_models[[mn]]$evaluation_metrics
        if (!is.null(evals) && is.null(evals$error)) {
          auroc_matrix[mn, i] <- evals$AUROC
        }
      } else {
        preds <- tryCatch({
          apply_dia(all_trained_models[[mn]]$model_object, datasets[[i]],
                    label_col_name = NULL, pos_class = new_positive_label, neg_class = new_negative_label)
        }, error = function(e) {
          warning(sprintf("Model %s failed on dataset %d", mn, i));
          NULL
        })

        if (!is.null(preds)) {
          evals <- tryCatch({
            evaluate_predictions_dia(preds, "f1", new_positive_label, new_negative_label)
          }, error = function(e) {
            warning(sprintf("Evaluation failed for %s on dataset %d", mn, i));
            NULL
          })

          if (!is.null(evals)) auroc_matrix[mn, i] <- evals$AUROC
        }
      }
    }
  }

  list(all_results = all_trained_models, auroc_matrix = auroc_matrix,
       model_categories = model_categories, dataset_names = dataset_names)
}



#' @title Imbalanced Data Diagnostic Modeling Pipeline
#' @description Extends \code{int_dia} by adding imbalance-specific models (EasyEnsemble).
#'   Produces a comprehensive set of models optimized for imbalanced datasets.
#'
#' @inheritParams int_dia
#'
#' @return Same structure as \code{int_dia} with additional imbalance-handling models.
#'
#' @examples
#' \dontrun{
#' imbalanced_results <- int_imbalance(train_imbalanced, test_imbalanced)
#' }
#'
#' @export
int_imbalance <- function(...,
                          model_names = NULL,
                          tune = TRUE,
                          n_estimators = 10,
                          seed = 123,
                          positive_label_value = 1,
                          negative_label_value = 0,
                          new_positive_label = "Positive",
                          new_negative_label = "Negative") {

  base_results <- int_dia(..., tune = tune, n_estimators = n_estimators, seed = seed,
                          positive_label_value = positive_label_value,
                          negative_label_value = negative_label_value,
                          new_positive_label = new_positive_label,
                          new_negative_label = new_negative_label)

  datasets <- list(...)
  train_data <- datasets[[1]]

  initialize_modeling_system_dia()

  if (is.null(model_names)) {
    all_base_models <- names(get_registered_models_dia())
  } else {
    all_base_models <- model_names
    registered <- names(get_registered_models_dia())
    invalid_models <- setdiff(all_base_models, registered)
    if (length(invalid_models) > 0) {
      stop(sprintf("The following models are not registered: %s\nRegistered models are: %s",
                   paste(invalid_models, collapse = ", "),
                   paste(registered, collapse = ", ")))
    }
  }
  imbalance_models <- list()
  imbalance_categories <- c()

  for (bm in all_base_models) {
    imb_name <- paste0("Imbalance_", bm)

    imb_result <- tryCatch({
      imbalance_dia(
        data = train_data, base_model_name = bm, n_estimators = n_estimators,
        tune_base_model = tune, threshold_choices = "youden",
        positive_label_value = positive_label_value, negative_label_value = negative_label_value,
        new_positive_label = new_positive_label, new_negative_label = new_negative_label, seed = seed
      )
    }, error = function(e) {
      warning(sprintf("Imbalance model '%s' failed: %s. Skipping this model.", imb_name, e$message))
      NULL
    })

    if (!is.null(imb_result)) {
      imbalance_models[[imb_name]] <- imb_result
      imbalance_categories[imb_name] <- "Imbalance"
    }
  }

  if (length(imbalance_models) == 0) {
    warning("All imbalance models failed to train. Returning only base results.")
    return(base_results)
  }

  all_models <- c(base_results$all_results, imbalance_models)
  all_categories <- c(base_results$model_categories, imbalance_categories)

  n_datasets <- length(datasets)
  auroc_extended <- matrix(NA, nrow = length(all_models), ncol = n_datasets,
                           dimnames = list(names(all_models), base_results$dataset_names))
  auroc_extended[rownames(base_results$auroc_matrix), ] <- base_results$auroc_matrix

  for (i in seq_along(datasets)) {
    for (mn in names(imbalance_models)) {
      preds <- tryCatch({
        apply_dia(imbalance_models[[mn]]$model_object, datasets[[i]],
                  label_col_name = NULL, pos_class = new_positive_label, neg_class = new_negative_label)
      }, error = function(e) { NULL })

      if (!is.null(preds)) {
        evals <- tryCatch({
          evaluate_predictions_dia(preds, "f1", new_positive_label, new_negative_label)
        }, error = function(e) { NULL })

        if (!is.null(evals)) auroc_extended[mn, i] <- evals$AUROC
      }
    }
  }

  list(all_results = all_models, auroc_matrix = auroc_extended,
       model_categories = all_categories, dataset_names = base_results$dataset_names)
}


# ==============================================================================
# Prognostic Model Integration
# ==============================================================================

#' @title Comprehensive Prognostic Modeling Pipeline
#' @description Executes a complete prognostic (survival) modeling workflow including
#'   single models, bagging, and stacking ensembles. Returns C-index and time-dependent
#'   AUROC metrics.
#'
#' @param ... Data frames for survival analysis. First = training; others = test sets.
#'   Format: first column = ID, second = outcome (0/1), third = time, remaining = features.
#' @param model_names Character vector specifying which models to use.
#'   If NULL (default), uses all registered prognostic models.
#' @param tune Logical, enable tuning. Default TRUE.
#' @param n_estimators Integer, bagging iterations. Default 10.
#' @param seed Integer for reproducibility. Default 123.
#' @param time_unit Time unit in data: "day", "month", or "year". Default "day".
#' @param years_to_evaluate Numeric vector of years for time-dependent AUROC. Default c(1,3,5).
#'
#' @return A list with:
#'   \itemize{
#'     \item \code{all_results}: All model outputs
#'     \item \code{cindex_matrix}: C-index values (models × datasets)
#'     \item \code{avg_auroc_matrix}: Average time-dependent AUROC (models × datasets)
#'     \item \code{model_categories}: Model category labels
#'     \item \code{dataset_names}: Dataset identifiers
#'   }
#'
#' @examples
#' \dontrun{
#' prognosis_results <- int_pro(train_pro, test_pro1, test_pro2)
#' }
#'
#' @export
int_pro <- function(...,
                    model_names = NULL,
                    tune = TRUE,
                    n_estimators = 10,
                    seed = 123,
                    time_unit = "day",
                    years_to_evaluate = c(1, 3, 5)) {

  datasets <- list(...)
  n_datasets <- length(datasets)
  if (n_datasets < 1) stop("At least one dataset required.")

  dataset_names <- c("Train", if (n_datasets > 1) paste0("Test", seq_len(n_datasets - 1)) else NULL)
  train_data <- datasets[[1]]

  initialize_modeling_system_pro()
  if (is.null(model_names)) {
    all_base_models <- names(get_registered_models_pro())
  } else {
    all_base_models <- model_names
    registered <- names(get_registered_models_pro())
    invalid_models <- setdiff(all_base_models, registered)
    if (length(invalid_models) > 0) {
      stop(sprintf("Invalid models: %s. Available: %s",
                   paste(invalid_models, collapse = ", "),
                   paste(registered, collapse = ", ")))
    }
  }

  all_trained_models <- list()
  model_categories <- c()

  set.seed(seed)
  single_results <- tryCatch({
    models_pro(
      data = train_data, model = all_base_models, tune = tune, seed = seed,
      time_unit = time_unit, years_to_evaluate = years_to_evaluate
    )
  }, error = function(e) {
    warning(sprintf("Single prognostic models training failed: %s. Skipping all single models.", e$message))
    list()
  })

  if (length(single_results) > 0) {
    all_trained_models <- c(all_trained_models, single_results)
    model_categories <- c(model_categories, setNames(rep("Single", length(single_results)), names(single_results)))
  }

  for (bm in all_base_models) {
    bag_name <- paste0("Bagging_", bm)

    bag_result <- tryCatch({
      bagging_pro(
        data = train_data, base_model_name = bm, n_estimators = n_estimators,
        subset_fraction = 0.632, tune_base_model = tune, time_unit = time_unit,
        years_to_evaluate = years_to_evaluate, seed = seed
      )
    }, error = function(e) {
      warning(sprintf("Bagging prognostic model '%s' failed: %s. Skipping.", bag_name, e$message))
      NULL
    })

    if (!is.null(bag_result)) {
      all_trained_models[[bag_name]] <- bag_result
      model_categories[bag_name] <- "Bagging"
    }
  }

  if (length(single_results) < 3) {
    warning("Not enough single models for stacking. Skipping all stacking models.")
  } else {
    cindices <- sapply(single_results, function(r) r$evaluation_metrics$C_index)
    top5 <- names(sort(cindices, decreasing = TRUE))[1:min(5, length(cindices))]
    top3 <- names(sort(cindices, decreasing = TRUE))[1:min(3, length(cindices))]

    total_stacking <- length(all_base_models) * 2
    current_stacking <- 0

    for (top_set in list(list(name = "5", models = top5), list(name = "3", models = top3))) {
      for (meta in all_base_models) {
        current_stacking <- current_stacking + 1
        stack_name <- paste0("Stacking(", top_set$name, ")_", meta)


        message(sprintf("Training stacking model %d/%d: %s",
                        current_stacking, total_stacking, stack_name))

        stack_result <- tryCatch({
          stacking_pro(
            results_all_models = single_results[top_set$models], data = train_data,
            meta_model_name = meta, top = length(top_set$models), tune_meta = tune,
            time_unit = time_unit, years_to_evaluate = years_to_evaluate, seed = seed
          )
        }, error = function(e) {
          warning(sprintf("Stacking prognostic model '%s' failed: %s. Skipping.", stack_name, e$message))
          NULL
        })

        if (!is.null(stack_result)) {
          all_trained_models[[stack_name]] <- stack_result
          model_categories[stack_name] <- paste0("Stacking(", top_set$name, ")")
        }
      }
    }
  }

  cindex_matrix <- matrix(NA, nrow = length(all_trained_models), ncol = n_datasets,
                          dimnames = list(names(all_trained_models), dataset_names))
  avg_auroc_matrix <- matrix(NA, nrow = length(all_trained_models), ncol = n_datasets,
                             dimnames = list(names(all_trained_models), dataset_names))

  for (i in seq_along(datasets)) {
    for (mn in names(all_trained_models)) {

      if (i == 1) {
        evals <- all_trained_models[[mn]]$evaluation_metrics
        if (!is.null(evals) && is.null(evals$error)) {
          cindex_matrix[mn, i] <- evals$C_index
          avg_auroc_matrix[mn, i] <- evals$AUROC_Average
        }
      } else {
        preds <- tryCatch({
          apply_pro(all_trained_models[[mn]]$model_object, datasets[[i]], time_unit = time_unit)
        }, error = function(e) {
          warning(sprintf("Model %s failed on dataset %d: %s", mn, i, e$message))
          NULL
        })

        if (!is.null(preds)) {
          evals <- tryCatch({
            evaluate_predictions_pro(preds, years_to_evaluate)
          }, error = function(e) {
            warning(sprintf("Evaluation failed for %s on dataset %d: %s", mn, i, e$message))
            NULL
          })

          if (!is.null(evals)) {
            cindex_matrix[mn, i] <- evals$C_index
            avg_auroc_matrix[mn, i] <- evals$AUROC_Average
          }
        }
      }
    }
  }

  list(all_results = all_trained_models, cindex_matrix = cindex_matrix,
       avg_auroc_matrix = avg_auroc_matrix, model_categories = model_categories,
       dataset_names = dataset_names)
}

# ==============================================================================
# Visualization Function
# ==============================================================================

#' @title Visualize Integrated Modeling Results
#' @description Creates a heatmap visualization with performance metrics across
#'   models and datasets, including category annotations and summary bar plots.
#'
#' @param results_obj Output from \code{int_dia}, \code{int_imbalance}, or \code{int_pro}.
#' @param metric_name Character string for metric used (e.g., "AUROC", "C-index").
#' @param output_file Optional file path to save plot. If NULL, plot is displayed.
#'
#' @return A ggplot object (invisibly).
#'
#' @examples
#' \dontrun{
#' results <- int_dia(train_dia, test_dia)
#' plot_integrated_results(results, "AUROC")
#' }
#'
#' @importFrom ggplot2 ggplot aes geom_tile geom_text scale_fill_gradientn labs theme_minimal
#'   theme element_text element_blank coord_fixed geom_col facet_grid scale_y_discrete
#' @importFrom dplyr mutate arrange desc
#' @importFrom tidyr pivot_longer
#' @importFrom cowplot get_legend
#' @importFrom patchwork plot_layout
#' @export
plot_integrated_results <- function(results_obj, metric_name = "AUROC", output_file = NULL) {

  if ("auroc_matrix" %in% names(results_obj)) {
    perf_matrix <- results_obj$auroc_matrix
  } else if ("cindex_matrix" %in% names(results_obj)) {
    perf_matrix <- results_obj$cindex_matrix
  } else {
    stop("Results object must contain 'auroc_matrix' or 'cindex_matrix'.")
  }

  categories <- results_obj$model_categories
  dataset_names <- results_obj$dataset_names

  test_cols <- if (ncol(perf_matrix) > 1) 2:ncol(perf_matrix) else 1
  avg_all <- rowMeans(perf_matrix, na.rm = TRUE)
  avg_test <- rowMeans(perf_matrix[, test_cols, drop = FALSE], na.rm = TRUE)

  order_idx <- order(avg_test, decreasing = TRUE)
  perf_matrix <- perf_matrix[order_idx, , drop = FALSE]
  categories <- categories[order_idx]
  avg_all <- avg_all[order_idx]
  avg_test <- avg_test[order_idx]

  df_heatmap <- as.data.frame(perf_matrix)
  df_heatmap$Model <- rownames(df_heatmap)
  df_heatmap$Category <- categories
  df_heatmap$Avg_All <- avg_all
  df_heatmap$Avg_Test <- avg_test

  df_long <- tidyr::pivot_longer(df_heatmap, cols = all_of(dataset_names),
                                 names_to = "Dataset", values_to = "Value")
  df_long$Model <- factor(df_long$Model, levels = rev(rownames(perf_matrix)))

  df_long$Dataset <- factor(df_long$Dataset, levels = dataset_names)

  color_palette <- c("#AAD09D", "#ECF4DD", "#FFF7AC", "#ECB477")

  p <- ggplot(df_long, aes(x = Dataset, y = Model, fill = Value)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.3f", Value)), size = 3, color = "black") +
    scale_fill_gradientn(colors = color_palette, na.value = "grey90",
                         name = metric_name, limits = range(perf_matrix, na.rm = TRUE)) +
    labs(title = paste(metric_name, "Heatmap Across Models and Datasets"),
         x = NULL, y = NULL) +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
          axis.text.y = element_text(face = "bold", size = 10),
          legend.position = "bottom",
          plot.title = element_text(hjust = 0.5, face = "bold"),
          panel.grid = element_blank(),
          plot.margin = margin(5, 5, 5, 5))

  df_bars <- data.frame(
    Model = factor(rep(rownames(perf_matrix), 2), levels = rev(rownames(perf_matrix))),
    Category = rep(categories, 2),
    Type = rep(c("All Datasets", "Test Datasets"), each = nrow(perf_matrix)),
    Average = c(avg_all, avg_test)
  )

  p_bars <- ggplot(df_bars, aes(x = Average, y = Model, fill = Category)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", Average)), hjust = -0.1, size = 2.5) +
    facet_grid(~ Type, scales = "free_x") +
    ggplot2::scale_x_continuous(
      expand = ggplot2::expansion(mult = c(0, 0.15))
    ) +
    scale_fill_manual(values = c("Single" = "#76c7c0", "Bagging" = "#f4a259",
                                 "Stacking(3)" = "#bc4b51","Stacking(10)" = "#bc4b51", "Stacking(5)" = "#5b8e7d",
                                 "Voting(10)" = "#8e7cc3", "Voting(5)" = "#c9a227",
                                 "Imbalance" = "#d4a5a5")) +
    labs(x = paste("Average", metric_name), y = NULL) +
    theme_minimal(base_size = 10) +
    theme(axis.text.y = element_blank(),
          legend.position = "bottom",
          strip.text = element_text(face = "bold"),
          plot.margin = margin(5, 5, 5, 5))


  p_bars_legend <- ggplot(df_bars, aes(x = Average, y = Model, fill = Category)) +
    geom_col(width = 0.7) +
    scale_fill_manual(values = c("Single" = "#76c7c0", "Bagging" = "#f4a259",
                                 "Stacking(3)" = "#bc4b51","Stacking(10)" = "#bc4b51", "Stacking(5)" = "#5b8e7d",
                                 "Voting(10)" = "#8e7cc3", "Voting(5)" = "#c9a227",
                                 "Imbalance" = "#d4a5a5"),
                      name = "Model Category") +
    theme(legend.position = "bottom",
          legend.box = "horizontal",
          legend.title = element_text(size = 10, face = "bold"),
          legend.text = element_text(size = 9))

  legend <- cowplot::get_legend(p_bars_legend)

  main_plot <- p + p_bars + patchwork::plot_layout(ncol = 2, widths = c(ncol(perf_matrix), 2))

  combined_plot <- main_plot / legend +
    patchwork::plot_layout(heights = c(20, 1))

  cell_width_mm <- 7
  cell_height_mm <- 28
  n_rows <- nrow(perf_matrix)
  n_cols <- ncol(perf_matrix)

  plot_width_inches <- (n_cols * cell_width_mm + 2 * cell_width_mm * 2) / 25.4
  plot_height_inches <- (n_rows * cell_height_mm + 20) / 25.4


  if (!is.null(output_file)) {
    ggplot2::ggsave(output_file, combined_plot,
                    width = 14, height = plot_height_inches + 1, dpi = 300)  # +1 for legend
  }
  if (!is.null(output_file)) {
    ggplot2::ggsave(output_file, combined_plot,
                    width = 14, height = max(8, nrow(perf_matrix) * 0.3), dpi = 300)
  }

  print(combined_plot)
  invisible(combined_plot)
}
