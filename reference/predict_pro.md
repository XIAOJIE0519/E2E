# Generic Prediction Interface for Prognostic Models

A unified S3 generic method to generate prognostic risk scores from
various trained model objects. This decouples the prediction
implementation from the high-level evaluation logic, facilitating
extensibility.

## Usage

``` r
predict_pro(object, newdata, ...)
```

## Arguments

- object:

  A trained model object with class `pro_model`.

- newdata:

  A data frame containing features for prediction.

- ...:

  Additional arguments passed to specific methods.

## Value

A numeric vector representing the prognostic risk score (higher values
typically indicate higher risk).
