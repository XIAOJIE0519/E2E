# 1. Getting Started

## E2E: An R Package for Easy-to-Build Ensemble Models

**E2E** is a comprehensive R package designed to streamline the
development, evaluation, and interpretation of machine learning models
for both **diagnostic (classification)** and **prognostic (survival
analysis)** tasks. It provides a robust, extensible framework for
training individual models and building powerful ensembles—including
Bagging, Voting, and Stacking—with minimal code. The package also
includes integrated tools for visualization and model explanation via
SHAP values.

**Author:** Shanjie Luan (ORCID: 0009-0002-8569-8526), Ximing Wang

**Citation:** If you use E2E in your research, please cite it as:
“Shanjie Luan, Ximing Wang (2025). E2E: An R Package for Easy-to-Build
Ensemble Models. <https://github.com/XIAOJIE0519/E2E>”

**Note:** The article is open source on CRAN and Github and is free to
use, but you have to cite our article if you use E2E in your research.
If you have any questions, please contact <Luan20050519@163.com>.

### Installation

The development version of E2E can be installed directly from GitHub
using `remotes`.

``` r
# If you don't have remotes, install it first:
# install.packages("remotes")
remotes::install_github("XIAOJIE0519/E2E")
```

After installation, load the package into your R session:

``` r
library(E2E)
```

### Core Concepts

E2E operates on two parallel tracks: **Diagnostic Models** and
**Prognostic Models**. Before using functions from either track, you
**must initialize** the corresponding system. This step registers a
suite of pre-defined, commonly used models.

#### Sample Data

To follow the examples, you’ll need sample data files. There are four
data frames included in the package for you to try: `train_dia`,
`test_dia`, `train_pro`, `test_pro`.

`train_dia` and `test_dia` are for diagnosis, with column names sample,
outcome, variable 1, 2, 3.

``` r
head(train_dia)
#>                         sample outcome AC009242.1 AC004637.1 AC246817.1
#> 1 TCGA-BH-A201-01A-11R-A14M-07       1     0.3988     1.3971     0.6180
#> 2 TCGA-C8-A12P-01A-11R-A115-07       1     0.0220     0.0000     0.0916
#> 3 TCGA-BH-A0W3-01A-11R-A109-07       1     0.0367     0.0000     0.0509
#> 4 TCGA-BH-A0H6-01A-21R-A056-07       1     0.1338     0.1563     0.0619
#> 5 TCGA-D8-A27V-01A-12R-A17B-07       1     0.1299     0.5056     0.0134
#> 6 TCGA-D8-A27M-01A-11R-A16F-07       1     0.1722     8.4883     0.2478
#>   AL139241.1 PRDM16.DT LINC01028 LINC00639 AL135841.1  HYMAI KCNIP2.AS1
#> 1     0.1441    0.2706    0.0000    0.2019     0.4235 0.0000     0.2728
#> 2     0.0530    0.0221    0.0000    0.0393     0.0000 0.0000     0.0502
#> 3     0.0883    0.0921    0.0000    0.2729     0.0000 0.0000     0.2508
#> 4     0.0000    0.2555    0.0000    0.0558     0.0000 0.0000     0.3662
#> 5     0.6259    0.1378    0.0349    0.2062     0.2299 0.0849     0.4279
#> 6     2.6272    0.1730    0.0926    0.4101     0.4063 0.0483     0.3490
#>   LINC00922 LINC01614 LINC01644 AC104237.3 AC016735.1 AC090125.1 AC008459.1
#> 1    0.8689   48.5811    4.1185     0.6571     4.9722     1.7074     0.5958
#> 2    1.6513   33.5412    0.0000     0.0000     0.0000     0.5024     0.0000
#> 3    2.3520   12.8974    0.1893     0.0000     0.0000     0.4186     0.0498
#> 4    1.8789   86.9410    0.0000     0.0000     0.0000     1.0694     0.0000
#> 5    0.6987   14.7284    3.5769     0.0951     1.1995     0.0824     0.0196
#> 6    0.9262   78.2584    0.0494     0.0000     6.3601     0.8736     0.0000
#>   LINC00958 AC112721.2 LINC00924
#> 1    8.5194     2.4894    0.3138
#> 2    0.3649     2.2892    0.1656
#> 3    0.2146     0.5722    0.1003
#> 4    0.0392     3.7589    0.2381
#> 5    0.8306     0.9759    0.4740
#> 6    0.1493     9.6523    0.1571
```

`train_pro` and `test_pro` are for prognosis, with column names sample,
outcome, time, variable 1, 2, 3.

``` r
head(train_pro)
#>                         sample outcome time LINC01497 LINC01028 AC084212.1
#> 1 TCGA-AC-A7VC-01A-11R-A352-07       0    1    0.0000         0     0.0000
#> 2 TCGA-C8-A275-01A-21R-A16F-07       0    1    0.0000         0     0.0000
#> 3 TCGA-C8-A1HJ-01A-11R-A13Q-07       0    5    0.1135         0     0.0298
#> 4 TCGA-PL-A8LX-01A-11R-A41B-07       0    5    0.0000         0     0.0000
#> 5 TCGA-AN-A041-01A-11R-A034-07       0    7    0.0000         0     0.0000
#> 6 TCGA-PL-A8LY-01A-11R-A41B-07       0    8    0.1910         0     0.0000
#>   AC104211.1 AL603840.1 AL590434.1 AC110772.2 LINC01614 C9orf163 AL391845.2
#> 1     0.4532     0.1294     0.0000     0.3130  224.7098   0.1968     1.1866
#> 2     0.0168     0.0000     0.0498     0.0000   25.0775   1.3142     0.4553
#> 3     0.0543     0.4426     0.0603     0.4218   11.4685   0.4421     0.2389
#> 4     0.5331     0.0000     0.0000     0.0000    4.8058   2.4996     2.9862
#> 5     0.3376     0.0000     0.0000     0.0343   77.7571   0.7764     1.7484
#> 6     0.1218     0.5118     0.9477     0.0526    0.0000   0.8931     0.0000
#>    HYMAI LINC01152 AL133467.1 LINC00165 LINC02408 AC092118.1 AP000851.2
#> 1 0.4287    0.4054     0.4250    0.1024    0.2572     0.0000     0.2342
#> 2 0.0421    0.0902     1.4662    0.1140    0.3021     0.0444     0.0651
#> 3 0.8610   33.7458     0.6109    0.0000    0.0642     0.2690     1.3673
#> 4 0.0000    0.7154     0.6136    0.0822    0.0458     0.0961     0.0000
#> 5 0.0166    0.1599     0.1118    0.0673    2.1791     0.0262     1.2314
#> 6 0.1780    3.3316     0.1714    0.9296    0.0000     0.8857     6.3743
#>   AC105046.1 LINC01929 AP001434.1 AC105219.1 AC133644.1 FAM153CP AC093515.1
#> 1     1.5372    0.7004     7.6471     0.0000     0.1563   0.0099     0.0000
#> 2     0.0940    1.7320     2.7820     0.1571     2.3483   0.1389     0.0526
#> 3     0.7133    0.3030     1.5853     0.0000     1.4043   0.0074     0.0000
#> 4     0.0135    0.2497     0.2359     0.4528     0.0000   0.1502     0.0253
#> 5     0.0222    1.1255     0.9668     0.1856     0.6166   2.6236     0.9121
#> 6     0.4428    0.0262     0.0000     0.2846     0.0000   0.0331     0.0636
#>   AC004990.1  HOTAIR AC055854.1 SIRLNT
#> 1     0.9918 11.2572     0.1926 0.0000
#> 2     0.0409  9.1333    10.0963 0.0714
#> 3     0.4125  4.0364     1.1249 0.0576
#> 4     0.0295  0.1821     0.0927 0.0343
#> 5     0.0000  0.1066     1.2157 0.0000
#> 6     0.0000  1.6027     0.9711 0.0000
```
