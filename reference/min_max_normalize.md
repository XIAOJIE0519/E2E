# Min-Max Normalization

Performs linear transformation of data to the range 0 to 1. Essential
for stacking ensembles to normalize risk scores from heterogeneous base
learners.

## Usage

``` r
min_max_normalize(x, min_val = NULL, max_val = NULL)
```

## Arguments

- x:

  A numeric vector.

- min_val:

  Optional reference minimum value (e.g., from training set).

- max_val:

  Optional reference maximum value (e.g., from training set).

## Value

A numeric vector of normalized values.
