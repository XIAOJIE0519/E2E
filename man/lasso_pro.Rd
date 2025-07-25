% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prognosis.R
\name{lasso_pro}
\alias{lasso_pro}
\title{Train a Lasso Cox Proportional Hazards Model}
\usage{
lasso_pro(X, y_surv, tune = FALSE)
}
\arguments{
\item{X}{A data frame of features.}

\item{y_surv}{A \code{survival::Surv} object representing the survival outcome.}

\item{tune}{Logical, whether to perform hyperparameter tuning (currently simplified/ignored
for direct \code{cv.glmnet} usage which inherently tunes lambda).}
}
\value{
A list of class "train" containing the trained \code{glmnet} model object,
names of features used in training, and model type. The returned object
also includes \code{fitted_scores} (linear predictor) and \code{y_surv}.
}
\description{
Trains a Cox proportional hazards model with Lasso regularization
using \code{glmnet}.
}
\examples{
\dontrun{
# Assuming `prepared_data` from load_and_prepare_data_pro example
# prepared_data <- load_and_prepare_data_pro(...)
# lasso_model <- lasso_pro(prepared_data$X, prepared_data$Y_surv)
# print(lasso_model$finalModel)
}
}
