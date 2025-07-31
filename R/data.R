# ==============================================================================
# data.R
# ------------------------------------------------------------------------------
#' @title Training Data for Diagnostic Models
#' @description A training dataset for diagnostic models, containing sample IDs,
#'   binary outcomes, and gene expression features.
#' @format A data frame with rows for samples and 22 columns:
#' \describe{
#'   \item{sample}{character. Unique identifier for each sample.}
#'   \item{outcome}{integer. The binary outcome, where 1 typically represents a positive case and 0 a negative case.}
#'   \item{AC004637.1}{numeric. Gene expression level.}
#'   \item{AC008459.1}{numeric. Gene expression level.}
#'   \item{AC009242.1}{numeric. Gene expression level.}
#'   \item{AC016735.1}{numeric. Gene expression level.}
#'   \item{AC090125.1}{numeric. Gene expression level.}
#'   \item{AC104237.3}{numeric. Gene expression level.}
#'   \item{AC112721.2}{numeric. Gene expression level.}
#'   \item{AC246817.1}{numeric. Gene expression level.}
#'   \item{AL135841.1}{numeric. Gene expression level.}
#'   \item{AL139241.1}{numeric. Gene expression level.}
#'   \item{HYMAI}{numeric. Gene expression level.}
#'   \item{KCNIP2.AS1}{numeric. Gene expression level.}
#'   \item{LINC00639}{numeric. Gene expression level.}
#'   \item{LINC00922}{numeric. Gene expression level.}
#'   \item{LINC00924}{numeric. Gene expression level.}
#'   \item{LINC00958}{numeric. Gene expression level.}
#'   \item{LINC01028}{numeric. Gene expression level.}
#'   \item{LINC01614}{numeric. Gene expression level.}
#'   \item{LINC01644}{numeric. Gene expression level.}
#'   \item{PRDM16.DT}{numeric. Gene expression level.}
#' }
#' @details This dataset is used to train machine learning models for diagnosis.
#'   The column names starting with 'AC', 'AL', 'LINC', etc., are feature variables.
#' @source Stored in `data/train_dia.rda`.
"train_dia"


#' @title Test Data for Diagnostic Models
#' @description A test dataset for evaluating diagnostic models, with a structure
#'   identical to `train_dia`.
#' @format A data frame with rows for samples and 22 columns:
#' \describe{
#'   \item{sample}{character. Unique identifier for each sample.}
#'   \item{outcome}{integer. The binary outcome (0 or 1).}
#'   \item{AC004637.1}{numeric. Gene expression level.}
#'   \item{AC008459.1}{numeric. Gene expression level.}
#'   \item{AC009242.1}{numeric. Gene expression level.}
#'   \item{AC016735.1}{numeric. Gene expression level.}
#'   \item{AC090125.1}{numeric. Gene expression level.}
#'   \item{AC104237.3}{numeric. Gene expression level.}
#'   \item{AC112721.2}{numeric. Gene expression level.}
#'   \item{AC246817.1}{numeric. Gene expression level.}
#'   \item{AL135841.1}{numeric. Gene expression level.}
#'   \item{AL139241.1}{numeric. Gene expression level.}
#'   \item{HYMAI}{numeric. Gene expression level.}
#'   \item{KCNIP2.AS1}{numeric. Gene expression level.}
#'   \item{LINC00639}{numeric. Gene expression level.}
#'   \item{LINC00922}{numeric. Gene expression level.}
#'   \item{LINC00924}{numeric. Gene expression level.}
#'   \item{LINC00958}{numeric. Gene expression level.}
#'   \item{LINC01028}{numeric. Gene expression level.}
#'   \item{LINC01614}{numeric. Gene expression level.}
#'   \item{LINC01644}{numeric. Gene expression level.}
#'   \item{PRDM16.DT}{numeric. Gene expression level.}
#' }
#' @source Stored in `data/test_dia.rda`.
"test_dia"


#' @title Training Data for Prognostic (Survival) Models
#' @description A training dataset for prognostic models, containing sample IDs,
#'   survival outcomes (time and event status), and gene expression features.
#' @format A data frame with rows for samples and 31 columns:
#' \describe{
#'   \item{sample}{character. Unique identifier for each sample.}
#'   \item{outcome}{integer. The event status, where 1 indicates an event occurred and 0 indicates censoring.}
#'   \item{time}{numeric. The time to event or censoring.}
#'   \item{AC004990.1}{numeric. Gene expression level.}
#'   \item{AC055854.1}{numeric. Gene expression level.}
#'   \item{AC084212.1}{numeric. Gene expression level.}
#'   \item{AC092118.1}{numeric. Gene expression level.}
#'   \item{AC093515.1}{numeric. Gene expression level.}
#'   \item{AC104211.1}{numeric. Gene expression level.}
#'   \item{AC105046.1}{numeric. Gene expression level.}
#'   \item{AC105219.1}{numeric. Gene expression level.}
#'   \item{AC110772.2}{numeric. Gene expression level.}
#'   \item{AC133644.1}{numeric. Gene expression level.}
#'   \item{AL133467.1}{numeric. Gene expression level.}
#'   \item{AL391845.2}{numeric. Gene expression level.}
#'   \item{AL590434.1}{numeric. Gene expression level.}
#'   \item{AL603840.1}{numeric. Gene expression level.}
#'   \item{AP000851.2}{numeric. Gene expression level.}
#'   \item{AP001434.1}{numeric. Gene expression level.}
#'   \item{C9orf163}{numeric. Gene expression level.}
#'   \item{FAM153CP}{numeric. Gene expression level.}
#'   \item{HOTAIR}{numeric. Gene expression level.}
#'   \item{HYMAI}{numeric. Gene expression level.}
#'   \item{LINC00165}{numeric. Gene expression level.}
#'   \item{LINC01028}{numeric. Gene expression level.}
#'   \item{LINC01152}{numeric. Gene expression level.}
#'   \item{LINC01497}{numeric. Gene expression level.}
#'   \item{LINC01614}{numeric. Gene expression level.}
#'   \item{LINC01929}{numeric. Gene expression level.}
#'   \item{LINC02408}{numeric. Gene expression level.}
#'   \item{SIRLNT}{numeric. Gene expression level.}
#' }
#' @details This dataset is used to train machine learning models for prognosis.
#'   The features are typically gene expression values.
#' @source Stored in `data/train_pro.rda`.
"train_pro"


#' @title Test Data for Prognostic (Survival) Models
#' @description A test dataset for evaluating prognostic models, with a structure
#'   identical to `train_pro`.
#' @format A data frame with rows for samples and 31 columns:
#' \describe{
#'   \item{sample}{character. Unique identifier for each sample.}
#'   \item{outcome}{integer. The event status (0 or 1).}
#'   \item{time}{numeric. The time to event or censoring.}
#'   \item{AC004990.1}{numeric. Gene expression level.}
#'   \item{AC055854.1}{numeric. Gene expression level.}
#'   \item{AC084212.1}{numeric. Gene expression level.}
#'   \item{AC092118.1}{numeric. Gene expression level.}
#'   \item{AC093515.1}{numeric. Gene expression level.}
#'   \item{AC104211.1}{numeric. Gene expression level.}
#'   \item{AC105046.1}{numeric. Gene expression level.}
#'   \item{AC105219.1}{numeric. Gene expression level.}
#'   \item{AC110772.2}{numeric. Gene expression level.}
#'   \item{AC133644.1}{numeric. Gene expression level.}
#'   \item{AL133467.1}{numeric. Gene expression level.}
#'   \item{AL391845.2}{numeric. Gene expression level.}
#'   \item{AL590434.1}{numeric. Gene expression level.}
#'   \item{AL603840.1}{numeric. Gene expression level.}
#'   \item{AP000851.2}{numeric. Gene expression level.}
#'   \item{AP001434.1}{numeric. Gene expression level.}
#'   \item{C9orf163}{numeric. Gene expression level.}
#'   \item{FAM153CP}{numeric. Gene expression level.}
#'   \item{HOTAIR}{numeric. Gene expression level.}
#'   \item{HYMAI}{numeric. Gene expression level.}
#'   \item{LINC00165}{numeric. Gene expression level.}
#'   \item{LINC01028}{numeric. Gene expression level.}
#'   \item{LINC01152}{numeric. Gene expression level.}
#'   \item{LINC01497}{numeric. Gene expression level.}
#'   \item{LINC01614}{numeric. Gene expression level.}
#'   \item{LINC01929}{numeric. Gene expression level.}
#'   \item{LINC02408}{numeric. Gene expression level.}
#'   \item{SIRLNT}{numeric. Gene expression level.}
#' }
#' @source Stored in `data/test_pro.rda`.
"test_pro"

