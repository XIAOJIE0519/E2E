% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prognosis.R
\name{stepcox_pro}
\alias{stepcox_pro}
\title{Train a Stepwise Cox Proportional Hazards Model}
\usage{
stepcox_pro(X, y_surv, tune = FALSE)
}
\arguments{
\item{X}{A data frame of features.}

\item{y_surv}{A \code{survival::Surv} object representing the survival outcome.}

\item{tune}{Logical, whether to perform hyperparameter tuning (currently ignored).}
}
\value{
A list of class "train" containing the trained \code{coxph} model object
after stepwise selection, names of features used in training, and model type.
The returned object also includes \code{fitted_scores} (linear predictor) and \code{y_surv}.
}
\description{
Trains a Cox proportional hazards model and performs backward
stepwise selection using \code{MASS::stepAIC} to select important features.
}
\examples{
\dontrun{
# Assuming `prepared_data` from load_and_prepare_data_pro example
# stepcox_model <- stepcox_pro(prepared_data$X, prepared_data$Y_surv)
# print(stepcox_model$finalModel)
}
}
