# Training Data for Diagnostic Models

A training dataset for diagnostic models, containing sample IDs, binary
outcomes, and gene expression features.

## Usage

``` r
train_dia
```

## Format

A data frame with rows for samples and 22 columns:

- sample:

  character. Unique identifier for each sample.

- outcome:

  integer. The binary outcome, where 1 typically represents a positive
  case and 0 a negative case.

- AC004637.1:

  numeric. Gene expression level.

- AC008459.1:

  numeric. Gene expression level.

- AC009242.1:

  numeric. Gene expression level.

- AC016735.1:

  numeric. Gene expression level.

- AC090125.1:

  numeric. Gene expression level.

- AC104237.3:

  numeric. Gene expression level.

- AC112721.2:

  numeric. Gene expression level.

- AC246817.1:

  numeric. Gene expression level.

- AL135841.1:

  numeric. Gene expression level.

- AL139241.1:

  numeric. Gene expression level.

- HYMAI:

  numeric. Gene expression level.

- KCNIP2.AS1:

  numeric. Gene expression level.

- LINC00639:

  numeric. Gene expression level.

- LINC00922:

  numeric. Gene expression level.

- LINC00924:

  numeric. Gene expression level.

- LINC00958:

  numeric. Gene expression level.

- LINC01028:

  numeric. Gene expression level.

- LINC01614:

  numeric. Gene expression level.

- LINC01644:

  numeric. Gene expression level.

- PRDM16.DT:

  numeric. Gene expression level.

## Source

Stored in `data/train_dia.rda`.

## Details

This dataset is used to train machine learning models for diagnosis. The
column names starting with 'AC', 'AL', 'LINC', etc., are feature
variables.
