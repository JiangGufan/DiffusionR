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


#' Train-Test-Validation split for synthetic distributions
#' @param X matrix of shape [n, d]
#' @param train_ratio proportion for training set
#' @param val_ratio proportion for validation set (test = 1 - train - val)
#' @param seed random seed for reproducibility
#' @return list with $train, $val, $test (each is a matrix)
#' @export
train_test_split <- function(X, train_ratio = 0.7, val_ratio = 0.15, seed = 42) {
  set.seed(seed)
  n <- nrow(X)
  idx <- sample(n)
  
  n_train <- floor(n * train_ratio)
  n_val <- floor(n * val_ratio)
  n_test <- n - n_train - n_val
  
  list(
    train = X[idx[1:n_train], , drop = FALSE],
    val = X[idx[(n_train + 1):(n_train + n_val)], , drop = FALSE],
    test = X[idx[(n_train + n_val + 1):n], , drop = FALSE]
  )
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

#' MNIST dataset with train/val/test split
#' @param train_ratio proportion for training (default 0.7)
#' @param val_ratio proportion for validation (default 0.15)
#' @param batch_size batch size for dataloaders
#' @param num_workers number of workers for data loading
#' @param seed random seed for reproducibility
#' @return list with $train_dl, $val_dl, $test_dl dataloaders
#' @export
mnist_split <- function(train_ratio = 0.7, val_ratio = 0.15, 
                        batch_size = 128, num_workers = 0L, seed = 42) {
  if (!requireNamespace("torchvision", quietly = TRUE)) {
    stop("Please install torchvision: remotes::install_github('mlverse/torchvision')")
  }
  
  set.seed(seed)
  torch::torch_manual_seed(seed)
  
  # Load full MNIST train set
  full_ds <- torchvision::mnist_dataset(
    root     = tempdir(),
    train    = TRUE,
    download = TRUE,
    transform = function(x) {
      torch::torch_tensor(as.array(x), dtype = torch::torch_float()) / 255
    }
  )
  
  # Get dataset size
  n <- length(full_ds)
  idx <- sample(n)
  
  n_train <- floor(n * train_ratio)
  n_val <- floor(n * val_ratio)
  
  train_idx <- idx[1:n_train]
  val_idx <- idx[(n_train + 1):(n_train + n_val)]
  test_idx <- idx[(n_train + n_val + 1):n]
  
  # Create subset datasets
  train_ds <- torch::dataset(
    name = "mnist_subset",
    initialize = function(full_dataset, indices) {
      self$full_ds <- full_dataset
      self$indices <- indices
    },
    .length = function() {
      length(self$indices)
    },
    .getitem = function(i) {
      self$full_ds[self$indices[i]]
    }
  )(full_ds, train_idx)
  
  val_ds <- torch::dataset(
    name = "mnist_subset",
    initialize = function(full_dataset, indices) {
      self$full_ds <- full_dataset
      self$indices <- indices
    },
    .length = function() {
      length(self$indices)
    },
    .getitem = function(i) {
      self$full_ds[self$indices[i]]
    }
  )(full_ds, val_idx)
  
  test_ds <- torch::dataset(
    name = "mnist_subset",
    initialize = function(full_dataset, indices) {
      self$full_ds <- full_dataset
      self$indices <- indices
    },
    .length = function() {
      length(self$indices)
    },
    .getitem = function(i) {
      self$full_ds[self$indices[i]]
    }
  )(full_ds, test_idx)
  
  # Create dataloaders
  list(
    train_dl = torch::dataloader(train_ds, batch_size = batch_size, shuffle = TRUE, num_workers = num_workers),
    val_dl = torch::dataloader(val_ds, batch_size = batch_size, shuffle = FALSE, num_workers = num_workers),
    test_dl = torch::dataloader(test_ds, batch_size = batch_size, shuffle = FALSE, num_workers = num_workers)
  )
}
