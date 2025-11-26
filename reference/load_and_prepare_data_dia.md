# Load and Prepare Data for Diagnostic Models

Loads a CSV file containing patient data, extracts features, and
converts the label column into a factor suitable for classification
models. Handles basic data cleaning like trimming whitespace and type
conversion.

## Usage

``` r
load_and_prepare_data_dia(
  data_path,
  label_col_name,
  positive_label_value = 1,
  negative_label_value = 0,
  new_positive_label = "Positive",
  new_negative_label = "Negative"
)
```

## Arguments

- data_path:

  A character string, the file path to the input CSV data. The first
  column is assumed to be a sample ID.

- label_col_name:

  A character string, the name of the column containing the class
  labels.

- positive_label_value:

  A numeric or character value that represents the positive class in the
  raw data.

- negative_label_value:

  A numeric or character value that represents the negative class in the
  raw data.

- new_positive_label:

  A character string, the desired factor level name for the positive
  class (e.g., "Positive").

- new_negative_label:

  A character string, the desired factor level name for the negative
  class (e.g., "Negative").

## Value

A list containing:

- `X`: A data frame of features (all columns except ID and label).

- `y`: A factor vector of class labels, with levels `new_negative_label`
  and `new_positive_label`.

- `sample_ids`: A vector of sample IDs (the first column of the input
  data).

- `pos_class_label`: The character string used for the positive class
  factor level.

- `neg_class_label`: The character string used for the negative class
  factor level.

- `y_original_numeric`: The original numeric/character vector of labels.

## Examples

``` r
# \donttest{
# Create a dummy CSV file in a temporary directory for demonstration
temp_csv_path <- tempfile(fileext = ".csv")
dummy_data <- data.frame(
  ID = paste0("Patient", 1:50),
  Disease_Status = sample(c(0, 1), 50, replace = TRUE),
  FeatureA = rnorm(50),
  FeatureB = runif(50, 0, 100),
  CategoricalFeature = sample(c("X", "Y", "Z"), 50, replace = TRUE)
)
write.csv(dummy_data, temp_csv_path, row.names = FALSE)

# Load and prepare data from the temporary file
prepared_data <- load_and_prepare_data_dia(
  data_path = temp_csv_path,
  label_col_name = "Disease_Status",
  positive_label_value = 1,
  negative_label_value = 0,
  new_positive_label = "Case",
  new_negative_label = "Control"
)

# Check prepared data structure
str(prepared_data$X)
#> 'data.frame':    50 obs. of  3 variables:
#>  $ FeatureA          : num  -1.1896 -0.5447 0.9136 0.045 0.0218 ...
#>  $ FeatureB          : num  46.9 47.1 28.1 46.3 95.2 ...
#>  $ CategoricalFeature: Factor w/ 3 levels "X","Y","Z": 3 1 1 1 1 2 1 3 1 3 ...
table(prepared_data$y)
#> 
#> Control    Case 
#>      32      18 

# Clean up the dummy file
unlink(temp_csv_path)
# }
```
