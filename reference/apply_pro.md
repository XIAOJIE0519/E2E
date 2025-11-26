# Apply Prognostic Model to New Data

Generates risk scores for new patients using a trained model.

## Usage

``` r
apply_pro(trained_model_object, new_data, time_unit = "day")
```

## Arguments

- trained_model_object:

  A trained object (class `pro_model`).

- new_data:

  Data frame of new patients.

- time_unit:

  Time unit for data preparation.

## Value

Data frame with IDs, outcomes (if available), and risk scores.
