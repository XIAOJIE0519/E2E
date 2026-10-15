library(E2E)

# Regression coverage for issues #1 and #4. Test with both legacy and current
# XGBoost: caret metadata must never be written into the booster itself.
set.seed(42)
X <- data.frame(a = rnorm(80), b = rnorm(80))
y <- factor(rep(c("Control", "Case"), 40), levels = c("Control", "Case"))

check_xb <- function(tune) {
  set.seed(123)
  fit <- xb_dia(X, y, tune = tune, cv_folds = 2, tune_length = 2)
  stopifnot(inherits(fit, "train"), all(is.finite(fit$results$ROC)),
            is.list(fit$finalModel), inherits(fit$finalModel$booster, "xgb.Booster"))
  prob <- predict(fit, X, type = "prob")
  stopifnot(identical(names(prob), levels(y)), nrow(prob) == nrow(X),
            all(is.finite(as.matrix(prob))), all(as.matrix(prob) >= 0),
            all(as.matrix(prob) <= 1), max(abs(rowSums(prob) - 1)) < 1e-7)
  native_prob <- predict(fit$finalModel$booster,
                         xgboost::xgb.DMatrix(as.matrix(X)))
  stopifnot(isTRUE(all.equal(prob$Control, as.numeric(native_prob))),
            identical(as.character(predict(fit, X)),
                      ifelse(prob$Control >= 0.5, "Control", "Case")))
  stopifnot(all(is.finite(caret::varImp(fit, scale = FALSE)$importance$Overall)))
  saved <- tempfile(fileext = ".rds")
  saveRDS(fit, saved)
  stopifnot(isTRUE(all.equal(prob, predict(readRDS(saved), X, type = "prob"))))
  unlink(saved)
  fit
}

fixed <- check_xb(FALSE)
tuned <- check_xb(TRUE)
stopifnot(nrow(tuned$results) == 2L)

# Exercise the PSOCK backend used by the diagnostic vignette and macOS CI.
if (requireNamespace("doParallel", quietly = TRUE)) {
  local({
    cl <- parallel::makePSOCKcluster(2)
    on.exit({parallel::stopCluster(cl); foreach::registerDoSEQ()})
    parallel::clusterCall(cl, function(paths) .libPaths(paths), .libPaths())
    doParallel::registerDoParallel(cl)
    parallel_fit <- check_xb(TRUE)
    stopifnot(nrow(parallel_fit$results) == 2L)
  })
}

bad_folds <- try(xb_dia(X, y, cv_folds = 41), silent = TRUE)
stopifnot(inherits(bad_folds, "try-error"))

# End-to-end diagnostic training and held-out evaluation retain class mapping
# and the threshold selected on the training data.
initialize_modeling_system_dia()
train <- data.frame(ID = seq_len(80), outcome = as.numeric(y == "Case"), X)
results <- suppressWarnings(models_dia(train, model = "xb", threshold_choices = 0.6))
stopifnot(is.null(results$xb$evaluation_metrics$error))
holdout <- train[1:20, ]
holdout$ID <- holdout$ID + 100L
holdout$a <- rev(holdout$a)
predictions <- apply_dia(results$xb$model_object, holdout, label_col_name = "outcome")
metrics <- suppressWarnings(evaluate_predictions_dia(
  predictions, threshold_choices = results$xb$evaluation_metrics$Final_Threshold
))
stopifnot(is.finite(metrics$AUROC), metrics$Threshold == 0.6)
summary_text <- capture.output(
  print_model_summary_dia("xb", list(evaluation_metrics = metrics), on_new_data = TRUE),
  type = "message"
)
stopifnot(any(grepl("0.6000", summary_text, fixed = TRUE)))
metrics_without_threshold <- metrics
metrics_without_threshold$Threshold <- NULL
invisible(capture.output(
  print_model_summary_dia("xb", list(evaluation_metrics = metrics_without_threshold),
                          on_new_data = TRUE), type = "message"
))

# Prognostic XGBoost must also avoid attaching fields to an immutable booster.
surv_y <- survival::Surv(seq_len(80) + 1, rep(c(1, 1, 0, 1), 20))
pro_fit <- xgb_pro(X, surv_y)
stopifnot(inherits(pro_fit$finalModel, "xgb.Booster"),
          length(pro_fit$fitted_scores) == nrow(X),
          all(is.finite(predict_pro(pro_fit, X))))
