% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/diagnosis.R
\name{lasso_dia}
\alias{lasso_dia}
\title{Train a Lasso (L1 Regularized Logistic Regression) Model for Classification}
\usage{
lasso_dia(X, y, tune = FALSE, cv_folds = 5)
}
\arguments{
\item{X}{A data frame of features.}

\item{y}{A factor vector of class labels.}

\item{tune}{Logical, whether to perform hyperparameter tuning for \code{lambda}
(if \code{TRUE}) or use a fixed value (if \code{FALSE}). \code{alpha} is fixed at 1 for Lasso.}

\item{cv_folds}{An integer, the number of cross-validation folds for \code{caret}.}
}
\value{
A \code{caret::train} object representing the trained Lasso model.
}
\description{
Trains a Lasso-regularized logistic regression model using \code{caret::train}
(via \code{glmnet} method) for binary classification.
}
\examples{
\dontrun{
# Assuming `prepared_data` from load_and_prepare_data_dia example
# lasso_model <- lasso_dia(prepared_data$X, prepared_data$y)
# print(lasso_model)
}
}
