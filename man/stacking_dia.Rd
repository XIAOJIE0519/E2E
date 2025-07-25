% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/diagnosis.R
\name{stacking_dia}
\alias{stacking_dia}
\title{Train a Stacking Diagnostic Model}
\usage{
stacking_dia(
  results_all_models,
  data_path,
  label_col_name,
  meta_model_name,
  top = 5,
  tune_meta = FALSE,
  threshold_choices = "f1",
  seed = 789,
  positive_label_value = 1,
  negative_label_value = 0,
  new_positive_label = "Positive",
  new_negative_label = "Negative"
)
}
\arguments{
\item{results_all_models}{A list of results from \code{run_models_dia()},
containing trained base model objects and their evaluation metrics.}

\item{data_path}{A character string, the file path to the input CSV data.
(Used to re-load and prepare original data for meta-model training).}

\item{label_col_name}{A character string, the name of the column containing
the class labels.}

\item{meta_model_name}{A character string, the name of the meta-model to use
(e.g., "lasso", "gbm"). This model must be registered.}

\item{top}{An integer, the number of top-performing base models (ranked by AUROC)
to select for the stacking ensemble.}

\item{tune_meta}{Logical, whether to enable tuning for the meta-model.}

\item{threshold_choices}{A character string (e.g., "f1", "youden", "default")
or a numeric value (0-1) for determining the evaluation threshold for the ensemble.}

\item{seed}{An integer, for reproducibility.}

\item{positive_label_value}{A numeric or character value in the raw data
representing the positive class.}

\item{negative_label_value}{A numeric or character value in the raw data
representing the negative class.}

\item{new_positive_label}{A character string, the desired factor level name
for the positive class (e.g., "Positive").}

\item{new_negative_label}{A character string, the desired factor level name
for the negative class (e.g., "Negative").}
}
\value{
A list containing:
\itemize{
\item \code{model_object}: A list describing the ensemble model, including
meta-model details and selected base models.
\item \code{sample_score}: A data frame with \code{sample} (ID), \code{label} (original numeric),
and final \code{score} (probability) from the stacking model.
\item \code{evaluation_metrics}: Performance metrics for the Stacking model.
}
}
\description{
Implements a Stacking ensemble for diagnostic models. It trains
multiple base models, then uses their predictions as features to train a
meta-model, which makes the final prediction. It selects top-performing
base models based on AUROC.
}
\examples{
\dontrun{
# 1. Create dummy data (same as for run_models_dia)
set.seed(123)
dummy_data <- data.frame(
  ID = paste0("Patient", 1:100),
  FeatureA = rnorm(100),
  FeatureB = runif(100, 0, 100),
  FeatureC = sample(c("X", "Y", "Z"), 100, replace = TRUE),
  Disease_Status = sample(c(0, 1), 100, replace = TRUE)
)
write.csv(dummy_data, "dummy_diagnosis_data.csv", row.names = FALSE)

# 2. Initialize the modeling system
initialize_modeling_system_dia()

# 3. Run a set of base models first
# base_model_results <- run_models_dia(
#   data_path = "dummy_diagnosis_data.csv",
#   label_col_name = "Disease_Status",
#   model = c("lasso", "ridge", "rf", "gbm", "dt"),
#   positive_label_value = 1,
#   negative_label_value = 0,
#   new_positive_label = "Case",
#   new_negative_label = "Control"
# )

# 4. Run Stacking with GBM as meta-model, using top 3 base models
# stacking_gbm_results <- stacking_dia(
#   results_all_models = base_model_results,
#   data_path = "dummy_diagnosis_data.csv",
#   label_col_name = "Disease_Status",
#   meta_model_name = "gbm",
#   top = 3,
#   threshold_choices = "f1",
#   positive_label_value = 1,
#   negative_label_value = 0,
#   new_positive_label = "Case",
#   new_negative_label = "Control"
# )
# print_model_summary_dia("Stacking (GBM)", stacking_gbm_results)

# 5. Clean up
# unlink("dummy_diagnosis_data.csv")
}
}
\seealso{
\code{\link{initialize_modeling_system_dia}}, \code{\link{register_model_dia}},
\code{\link{run_models_dia}}, \code{\link{load_and_prepare_data_dia}},
\code{\link{evaluate_model_dia}}
}
