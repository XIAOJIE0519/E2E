% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prognosis.R
\name{evaluate_model_pro}
\alias{evaluate_model_pro}
\title{Evaluate Prognostic Model Performance}
\usage{
evaluate_model_pro(
  trained_model_obj = NULL,
  X_data = NULL,
  Y_surv_obj,
  sample_ids,
  years_to_evaluate = c(1, 3, 5),
  precomputed_score = NULL,
  meta_normalize_params = NULL
)
}
\arguments{
\item{trained_model_obj}{A trained model object (of class "train" as returned
by model training functions like \code{lasso_pro}, \code{rsf_pro}, etc.). Can be \code{NULL}
if \code{precomputed_score} is provided.}

\item{X_data}{A data frame of features corresponding to the data used for evaluation.
Required if \code{trained_model_obj} is provided and \code{precomputed_score} is \code{NULL}.}

\item{Y_surv_obj}{A \code{survival::Surv} object for the evaluation data.}

\item{sample_ids}{A vector of sample IDs for the evaluation data.}

\item{years_to_evaluate}{A numeric vector of specific years at which to
calculate time-dependent AUROC.}

\item{precomputed_score}{Optional. A numeric vector of precomputed prognostic
scores for the samples. If provided, \code{trained_model_obj} and \code{X_data} are
not strictly necessary for score derivation.}

\item{meta_normalize_params}{Optional. A list of normalization parameters
(min/max values) used for base model scores in a stacking ensemble.
Used when \code{trained_model_obj} is a stacking model to ensure consistent
normalization during prediction.}
}
\value{
A list containing:
\itemize{
\item \code{sample_score}: A data frame with \code{ID}, \code{outcome}, \code{time}, and \code{score} columns.
\item \code{evaluation_metrics}: A list of performance metrics:
\itemize{
\item \code{C_index}: Harrell's C-index.
\item \code{AUROC_Years}: A named list of time-dependent AUROC values for specified years.
\item \code{AUROC_Average}: The average of time-dependent AUROC values.
\item \code{KM_HR}: Hazard Ratio for High vs Low risk groups (split by median score).
\item \code{KM_P_value}: P-value for the KM group comparison.
\item \code{KM_Cutoff}: The score cutoff used to define High/Low risk groups (median score).
}
}
}
\description{
Evaluates the performance of a trained prognostic model using
various metrics relevant to survival analysis, including C-index,
time-dependent AUROC, and Kaplan-Meier (KM) group analysis (Hazard Ratio and p-value).
}
\examples{
\dontrun{
# Assuming `prepared_data` from load_and_prepare_data_pro example
# And a trained model, e.g., lasso_model <- lasso_pro(prepared_data$X, prepared_data$Y_surv)
#
# Evaluate the model
# eval_results <- evaluate_model_pro(
#   trained_model_obj = lasso_model,
#   X_data = prepared_data$X,
#   Y_surv_obj = prepared_data$Y_surv,
#   sample_ids = prepared_data$sample_ids,
#   years_to_evaluate = c(1, 2, 3)
# )
# str(eval_results)
}
}
