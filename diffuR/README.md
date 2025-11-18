# diffuR

Minimal Diffusion Models in R with Rcpp and torch.

## Install (from source)

```r
# In R:
install.packages("Rcpp"); install.packages("RcppArmadillo")
install.packages("torch"); install.packages("magrittr"); install.packages("ggplot2")
install.packages("foreach"); install.packages("doParallel")
# torchvision for MNIST dataloader:
# remotes::install_github("mlverse/torchvision")

# Build & install this package (assuming you unzipped to some path):
devtools::install_local("path/to/diffuR")
```

## Quickstart

See `vignettes/getting_started.Rmd` or `inst/benchmarks/`.
