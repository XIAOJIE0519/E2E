# Training Data for Prognostic (Survival) Models

A training dataset for prognostic models, containing sample IDs,
survival outcomes (time and event status), and gene expression features.

## Usage

``` r
train_pro
```

## Format

A data frame with rows for samples and 31 columns:

- sample:

  character. Unique identifier for each sample.

- outcome:

  integer. The event status, where 1 indicates an event occurred and 0
  indicates censoring.

- time:

  numeric. The time to event or censoring.

- AC004990.1:

  numeric. Gene expression level.

- AC055854.1:

  numeric. Gene expression level.

- AC084212.1:

  numeric. Gene expression level.

- AC092118.1:

  numeric. Gene expression level.

- AC093515.1:

  numeric. Gene expression level.

- AC104211.1:

  numeric. Gene expression level.

- AC105046.1:

  numeric. Gene expression level.

- AC105219.1:

  numeric. Gene expression level.

- AC110772.2:

  numeric. Gene expression level.

- AC133644.1:

  numeric. Gene expression level.

- AL133467.1:

  numeric. Gene expression level.

- AL391845.2:

  numeric. Gene expression level.

- AL590434.1:

  numeric. Gene expression level.

- AL603840.1:

  numeric. Gene expression level.

- AP000851.2:

  numeric. Gene expression level.

- AP001434.1:

  numeric. Gene expression level.

- C9orf163:

  numeric. Gene expression level.

- FAM153CP:

  numeric. Gene expression level.

- HOTAIR:

  numeric. Gene expression level.

- HYMAI:

  numeric. Gene expression level.

- LINC00165:

  numeric. Gene expression level.

- LINC01028:

  numeric. Gene expression level.

- LINC01152:

  numeric. Gene expression level.

- LINC01497:

  numeric. Gene expression level.

- LINC01614:

  numeric. Gene expression level.

- LINC01929:

  numeric. Gene expression level.

- LINC02408:

  numeric. Gene expression level.

- SIRLNT:

  numeric. Gene expression level.

## Source

Stored in `data/train_pro.rda`.

## Details

This dataset is used to train machine learning models for prognosis. The
features are typically gene expression values.
