# Swiss roll / Mixture of Gaussians
#' @export
swiss_roll <- function(n = 2000, noise = 0.1, seed = 1){
  set.seed(seed)
  t <- (3*pi/2) * (1 + 2*runif(n))
  x <- cbind(t*cos(t), t*sin(t))/ (3*pi) + matrix(rnorm(2*n, sd = noise), n, 2)
  colnames(x) <- c("x1","x2")
  x
}

# 8-Gaussians ring
#' @export
gauss8_ring <- function(n = 2000, radius = 2, noise = 0.1, seed = 1) {
  set.seed(seed)
  K <- 8
  # 每个簇中心的角度
  angles <- seq(0, 2*pi, length.out = K + 1)[- (K + 1)]
  
  # 给每个样本随机分配一个簇
  k_id <- sample.int(K, n, replace = TRUE)
  mu_x <- radius * cos(angles[k_id])
  mu_y <- radius * sin(angles[k_id])
  
  x <- cbind(
    mu_x + rnorm(n, sd = noise),
    mu_y + rnorm(n, sd = noise)
  )
  colnames(x) <- c("x1", "x2")
  x
}

# 爱心形状（parametric heart curve）
heart2d <- function(n = 2000, noise = 0.1, seed = 1) {
  set.seed(seed)
  t <- runif(n, 0, 2*pi)

  x <- 16 * sin(t)^3
  y <- 13 * cos(t) - 5 * cos(2*t) - 2 * cos(3*t) - cos(4*t)

  # 缩放到大约 [-2,2] 范围
  x <- x / 8
  y <- y / 8

  x <- x + rnorm(n, sd = noise)
  y <- y + rnorm(n, sd = noise)

  out <- cbind(x, y)
  colnames(out) <- c("x1", "x2")
  out
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

# # MNIST dataloader
# mnist_train_dataloader <- function(batch_size = 128){
#   if(!requireNamespace("torchvision", quietly = TRUE)){
#     stop("Please install torchvision: remotes::install_github('mlverse/torchvision')")
#   }
#   ds <- torchvision::mnist_dataset(
#     root = tempdir(), train = TRUE, download = TRUE,
#     transform = function(x){
#       x <- torch::torch_tensor(as.array(x), dtype = torch::torch_float())/255
#       # x$unsqueeze(1L)  # [1,28,28]
#       # 不再 unsqueeze，在这里保持 [28,28]
#       x
#     }
#   )
#   torch::dataloader(ds, batch_size = batch_size, shuffle = TRUE)
# }

# MNIST dataloader（支持大 batch + 多 worker）
mnist_train_dataloader <- function(batch_size = 128,
                                   num_workers = 0L) {
  if (!requireNamespace("torchvision", quietly = TRUE)) {
    stop("Please install torchvision: remotes::install_github('mlverse/torchvision')")
  }

  ds <- torchvision::mnist_dataset(
    root     = tempdir(),
    train    = TRUE,
    download = TRUE,
    transform = function(x) {
      x <- torch::torch_tensor(as.array(x), dtype = torch::torch_float()) / 255
      # 保持 [28, 28]，跟你原来的设定一致
      x
    }
  )

  torch::dataloader(
    ds,
    batch_size  = batch_size,
    shuffle     = TRUE,
    num_workers = num_workers  # ✅ 这里开多 worker
  )
}
