# Swiss roll / Mixture of Gaussians
#' @export
swiss_roll <- function(n = 2000, noise = 0.1, seed = 1){
  set.seed(seed)
  t <- (3*pi/2) * (1 + 2*runif(n))
  x <- cbind(t*cos(t), t*sin(t))/ (3*pi) + matrix(rnorm(2*n, sd = noise), n, 2)
  colnames(x) <- c("x1","x2")
  x
}

#' @export
plot_2d_samples <- function(real, fake){
  df1 <- data.frame(real); df1$type <- "real"
  df2 <- data.frame(fake); df2$type <- "fake"
  df <- rbind(df1, df2)
  ggplot2::ggplot(df, ggplot2::aes(x=x1, y=x2, color=type)) +
    ggplot2::geom_point(alpha=0.6, size=1) +
    ggplot2::coord_equal() + ggplot2::theme_minimal()
}

# MNIST dataloader
mnist_train_dataloader <- function(batch_size = 128){
  if(!requireNamespace("torchvision", quietly = TRUE)){
    stop("Please install torchvision: remotes::install_github('mlverse/torchvision')")
  }
  ds <- torchvision::mnist_dataset(
    root = tempdir(), train = TRUE, download = TRUE,
    transform = function(x){
      x <- torch::torch_tensor(as.array(x), dtype = torch::torch_float())/255
      # x$unsqueeze(1L)  # [1,28,28]
      # 不再 unsqueeze，在这里保持 [28,28]
      x
    }
  )
  torch::dataloader(ds, batch_size = batch_size, shuffle = TRUE)
}
