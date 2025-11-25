#' @export
psnr <- function(img, ref, max_val = 1){
  # img, ref: arrays [B,H,W] in [0,1]
  mse <- mean((img - ref)^2)
  10 * log10(max_val^2 / mse)
}

#' @export
ssim <- function(img, ref, C1 = 0.01^2, C2 = 0.03^2){
  # simplified SSIM over whole image (not windowed), for demonstration
  mu_x <- mean(img); mu_y <- mean(ref)
  sigma_x2 <- mean((img - mu_x)^2)
  sigma_y2 <- mean((ref - mu_y)^2)
  sigma_xy <- mean((img - mu_x)*(ref - mu_y))
  ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)*(sigma_x2 + sigma_y2 + C2))
}
#' Maximum Mean Discrepancy (MMD) for distribution comparison
#' Measures distance between two distributions using kernel trick
#' @param X1 matrix of shape [n1, d], samples from distribution 1
#' @param X2 matrix of shape [n2, d], samples from distribution 2
#' @param kernel_type 'rbf' or 'poly' (currently only 'rbf' used)
#' @param sigma bandwidth for RBF kernel
#' @return MMD value (0 = identical distributions)
#' @export
mmd <- function(X1, X2, kernel_type = "rbf", sigma = 1.0) {
  X1 <- as.matrix(X1)
  X2 <- as.matrix(X2)
  n1 <- nrow(X1)
  n2 <- nrow(X2)
  
  if (n1 < 2 || n2 < 2) {
    stop("Need at least 2 samples in each set for MMD.")
  }
  
  # Compute kernel matrix on [X1; X2]
  rbf_kernel <- function(X, Y, sigma) {
    # Gaussian kernel: exp(-||x-y||^2 / (2*sigma^2))
    D <- as.matrix(dist(rbind(X, Y)))
    K <- exp(-D^2 / (2 * sigma^2))
    K
  }
  
  K <- rbf_kernel(X1, X2, sigma)
  K11 <- K[1:n1, 1:n1, drop = FALSE]
  K22 <- K[(n1 + 1):(n1 + n2), (n1 + 1):(n1 + n2), drop = FALSE]
  K12 <- K[1:n1, (n1 + 1):(n1 + n2), drop = FALSE]
  
  # ✅ 无偏 MMD^2:
  # 1/(n1(n1-1)) Σ_{i≠j} k(x_i,x_j) + 1/(n2(n2-1)) Σ_{i≠j} k(y_i,y_j)
  #   - 2/(n1 n2) Σ_{ij} k(x_i,y_j)
  term_x  <- (sum(K11) - sum(diag(K11))) / (n1 * (n1 - 1))
  term_y  <- (sum(K22) - sum(diag(K22))) / (n2 * (n2 - 1))
  term_xy <- 2 * mean(K12)
  
  mmd_sq <- term_x + term_y - term_xy
  sqrt(max(mmd_sq, 0))  # Ensure non-negative
}


#' Frechet Distance for distribution comparison
#' Simpler alternative to FID for low-dimensional data
#' @param X_real matrix of shape [n, d], real samples
#' @param X_fake matrix of shape [n, d], generated samples
#' @return Frechet distance
#' @export
frechet_distance <- function(X_real, X_fake) {
  # Compute mean and covariance
  mu_real <- colMeans(X_real)
  mu_fake <- colMeans(X_fake)
  
  Sigma_real <- cov(X_real)
  Sigma_fake <- cov(X_fake)
  
  # Frechet distance: ||mu_real - mu_fake||^2 + trace(Sigma_real + Sigma_fake - 2*sqrt(Sigma_real*Sigma_fake))
  mean_diff <- sum((mu_real - mu_fake)^2)
  
  # For covariance term, use spectral decomposition
  # sqrt(Sigma_real * Sigma_fake) = U * diag(sqrt(lambda)) * U^T
  sqrt_sigma_real <- with(eigen(Sigma_real), vectors %*% diag(sqrt(pmax(values, 0))) %*% t(vectors))
  prod <- sqrt_sigma_real %*% Sigma_fake %*% sqrt_sigma_real
  sqrt_prod <- with(eigen(prod), vectors %*% diag(sqrt(pmax(values, 0))) %*% t(vectors))
  
  cov_diff <- sum(diag(Sigma_real + Sigma_fake - 2 * sqrt_prod))
  
  sqrt(mean_diff + cov_diff)
}

#' Wasserstein Distance (1D approximation)
#' For efficient computation in low dimensions
#' @param X_real vector or matrix [n], real samples
#' @param X_fake vector or matrix [n], generated samples
#' @return Wasserstein distance
#' @export
wasserstein_1d <- function(X_real, X_fake) {
  # Sort samples
  x_real_sorted <- sort(as.numeric(X_real))
  x_fake_sorted <- sort(as.numeric(X_fake))
  
  # Pad shorter sequence with its last value
  if (length(x_real_sorted) > length(x_fake_sorted)) {
    x_fake_sorted <- c(x_fake_sorted, rep(x_fake_sorted[length(x_fake_sorted)], 
                                          length(x_real_sorted) - length(x_fake_sorted)))
  } else if (length(x_fake_sorted) > length(x_real_sorted)) {
    x_real_sorted <- c(x_real_sorted, rep(x_real_sorted[length(x_real_sorted)], 
                                          length(x_fake_sorted) - length(x_real_sorted)))
  }
  
  mean(abs(x_real_sorted - x_fake_sorted))
}

#' Coverage and Density metrics for generative models
#' Evaluates both diversity and quality of samples
#' @param X_real matrix [n_real, d], real samples
#' @param X_fake matrix [n_fake, d], generated samples
#' @param k number of nearest neighbors for radius estimation / density
#' @param q quantile used to define real-data radius (e.g. 0.5 => median)
#' @return list with coverage, density, coverage_percent, radius_thr
#' @export
coverage_density <- function(X_real, X_fake, k = 5, q = 0.5) {
  X_real <- as.matrix(X_real)
  X_fake <- as.matrix(X_fake)
  n_real <- nrow(X_real)
  n_fake <- nrow(X_fake)
  
  if (n_real < 2 || n_fake < 1) {
    stop("Need at least 2 real samples and 1 fake sample.")
  }
  
  # Compute pairwise distances on concatenated data
  D <- as.matrix(dist(rbind(X_real, X_fake)))
  D_rr <- D[1:n_real, 1:n_real, drop = FALSE]                     # real -> real
  D_rf <- D[1:n_real, (n_real + 1):(n_real + n_fake), drop = FALSE] # real -> fake
  D_fr <- D[(n_real + 1):(n_real + n_fake), 1:n_real, drop = FALSE] # fake -> real
  
  # --- 1) 在真实数据中定义尺度：每个真实点到第 (k+1) 个最近“真实点”的距离 ---
  #     （+1 是为了跳过自身那一个 0 距离）
  rr_knn <- apply(D_rr, 1, function(d) {
    d_sorted <- sort(d)
    idx <- min(k + 1, length(d_sorted))
    d_sorted[idx]
  })
  radius_thr <- as.numeric(quantile(rr_knn, q))
  
  # --- 2) Coverage: 真实点能否被假样本“覆盖” ---
  #     若某个真实点的最近假样本距离 <= radius_thr，则视作被覆盖
  min_rf <- apply(D_rf, 1, min)
  coverage <- mean(min_rf <= radius_thr)
  
  # --- 3) Density: 假样本到最近真实样本的平均距离（越小越好） ---
  nn_dist_fr <- apply(D_fr, 1, min)
  density <- mean(nn_dist_fr)
  
  list(
    coverage          = coverage,
    density           = density,
    coverage_percent  = coverage * 100,
    radius_thr        = radius_thr
  )
}


#' Evaluate generated samples against real samples (comprehensive)
#' @param X_real matrix [n_real, d], real samples (for distribution metrics)
#' @param X_fake matrix [n_fake, d], generated samples
#' @param type 'distribution' for MLP, 'image' for CNN
#' @param verbose print results
#' @return list with all metrics
#' @export
evaluate_samples <- function(X_real, X_fake, type = "distribution", verbose = TRUE) {
  
  stopifnot(ncol(X_real) == ncol(X_fake))
  
  if (type == "distribution") {
    # For low-dimensional distributions
    X_real <- as.matrix(X_real)
    X_fake <- as.matrix(X_fake)
    
    mmd_val  <- mmd(X_real, X_fake, sigma = 1.0)
    fd       <- frechet_distance(X_real, X_fake)
    w1d      <- wasserstein_1d(X_real[, 1], X_fake[, 1])
    cov_dens <- coverage_density(X_real, X_fake, k = 5, q = 0.5)
    
    metrics <- list(
      mmd            = mmd_val,
      frechet        = fd,
      wasserstein_1d = w1d,
      coverage       = cov_dens$coverage,
      density        = cov_dens$density
    )
    
    if (verbose) {
      cat("\n===== Distribution Evaluation Metrics =====\n")
      cat(sprintf("MMD:                %.6f (lower better)\n", mmd_val))
      cat(sprintf("Frechet Distance:   %.6f (lower better)\n", fd))
      cat(sprintf("1D Wasserstein:     %.6f (lower better)\n", w1d))
      cat(sprintf("Coverage:           %.2f%% (higher better)\n", cov_dens$coverage_percent))
      cat(sprintf("Density (NN dist):  %.6f (lower better)\n", cov_dens$density))
      cat("\n")
    }
    
  } else if (type == "image") {
    # For images, use different metrics
    # Flatten images for distribution comparison
    X_real_flat <- matrix(X_real, nrow = dim(X_real)[1])
    X_fake_flat <- matrix(X_fake, nrow = dim(X_fake)[1])
    
    mmd_val  <- mmd(X_real_flat, X_fake_flat, sigma = 0.5)
    cov_dens <- coverage_density(X_real_flat, X_fake_flat, k = 3, q = 0.5)
    
    metrics <- list(
      mmd      = mmd_val,
      coverage = cov_dens$coverage,
      density  = cov_dens$density
    )
    
    if (verbose) {
      cat("\n===== Image Evaluation Metrics =====\n")
      cat(sprintf("MMD (flattened):    %.6f (lower better)\n", mmd_val))
      cat(sprintf("Coverage:           %.2f%% (higher better)\n", cov_dens$coverage_percent))
      cat(sprintf("Density (NN dist):  %.6f (lower better)\n", cov_dens$density))
      cat("\n")
    }
  }
  
  invisible(metrics)
}
