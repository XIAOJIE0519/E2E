% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prognosis.R
\name{gbm_pro}
\alias{gbm_pro}
\title{Train a Gradient Boosting Machine (GBM) for Survival Data}
\usage{
gbm_pro(X, y_surv, tune = FALSE)
}
\arguments{
\item{X}{A data frame of features.}

\item{y_surv}{A \code{survival::Surv} object representing the survival outcome.}

\item{tune}{Logical, whether to perform simplified hyperparameter tuning.
If \code{TRUE}, \code{n.trees}, \code{interaction.depth}, and \code{shrinkage} are set to
predefined values suitable for tuning; otherwise, default values are used.}
}
\value{
A list of class "train" containing the trained \code{gbm} model object,
names of features used in training, and model type. The returned object
also includes \code{fitted_scores} (linear predictor), \code{y_surv}, and \code{best_iter}.
}
\description{
Trains a Gradient Boosting Machine (GBM) model with a Cox
proportional hazards loss function using \code{gbm}.
}
\examples{
\dontrun{
# Assuming `prepared_data` from load_and_prepare_data_pro example
# gbm_model <- gbm_pro(prepared_data$X, prepared_data$Y_surv)
# print(gbm_model$finalModel)
}
}
