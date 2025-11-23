
---

# E2E: An R Package for Easy-to-Build Ensemble Models

<!-- badges: start -->
[![R-CMD-check](https://github.com/XIAOJIE0519/E2E/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/XIAOJIE0519/E2E/actions/workflows/R-CMD-check.yaml)
[![pkgdown](https://github.com/XIAOJIE0519/E2E/actions/workflows/pkgdown.yaml/badge.svg)](https://github.com/XIAOJIE0519/E2E/actions/workflows/pkgdown.yaml)
<!-- badges: end -->

**E2E** is a comprehensive R package designed to streamline the development, evaluation, and interpretation of machine learning models for both **diagnostic (classification)** and **prognostic (survival analysis)** tasks. It provides a robust, extensible framework for training individual models and building powerful ensembles—including Bagging, Voting, and Stacking—with minimal code. The package also includes integrated tools for visualization and model explanation via SHAP values.

**Author:** Shanjie Luan (ORCID: 0009-0002-8569-8526, First and Corresponding Author), Ximing Wang

**Citation:** If you use E2E in your research, please cite it as:
"Luan, S. and Wang, X. (2025), E2E: An R Package for Easy-to-Build Ensemble Models. Med Research. [https://doi.org/10.1002/mdr2.70030](https://doi.org/10.1002/mdr2.70030)"

**Note:** The article is open source on CRAN and Github and is free to use, but you have to cite our article if you use E2E in your research. If you have any questions, please contact [Luan20050519@163.com](mailto:Luan20050519@163.com).

## Documentation

**For complete documentation, tutorials, and function references, please visit our pkgdown website:**

**[https://XIAOJIE0519.github.io/E2E/](https://XIAOJIE0519.github.io/E2E/)**

**back to our github website:**

**[https://github.com/XIAOJIE0519/E2E](https://github.com/XIAOJIE0519/E2E)**

---

## Installation

The development version of E2E can be installed directly from GitHub using `remotes`.

```R
# If you don't have remotes, install it first:
# install.packages("remotes")
remotes::install_github("XIAOJIE0519/E2E")
```

After installation, load the package into your R session:

```R
library(E2E)
```

## Methodological Framework
![Workflow](https://github.com/user-attachments/assets/6a908218-f84d-4b40-83ed-a6c6acb0fe37)

