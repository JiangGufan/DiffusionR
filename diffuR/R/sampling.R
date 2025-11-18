#' DDPM sampling for distribution (MLP) or image (CNN)
#' @export
# sample_ddpm <- function(fit, n = 64, steps = NULL, shape_img = NULL, seed = 123){
#   set.seed(seed); torch::torch_manual_seed(seed)
#   if(is.null(steps)) steps <- fit$T
#   sch <- fit$schedule
#   if(fit$type == "ddpm_mlp"){
#     d <- fit$dim
#     x <- torch::torch_randn(n, d)
#     for (ti in steps:1){
#       t_frac <- torch::torch_full(c(n,1), ti/fit$T)
#       eps <- fit$model(x, t_frac)
#       sa_t   <- torch::torch_tensor(matrix(sch$sqrt_alpha_bar[ti], n, 1))
#       sa_tm1 <- torch::torch_tensor(matrix(ifelse(ti>1, sch$sqrt_alpha_bar[ti-1], 1), n, 1))
#       om_t   <- torch::torch_tensor(matrix(1 - sch$alpha_bar[ti], n, 1))
#       om_tm1 <- torch::torch_tensor(matrix(ifelse(ti>1, 1 - sch$alpha_bar[ti-1], 0), n, 1))
#       beta_t <- torch::torch_tensor(matrix(sch$beta[ti], n, 1))

#       x0_hat <- x / sa_t - torch::torch_sqrt(om_t) * eps
#       mu <- (sa_tm1 * beta_t / om_t) * x0_hat +
#             (torch::torch_sqrt(1-beta_t) * (om_tm1 / om_t)) * x
#       if (ti > 1){
#         var <- (om_tm1 / om_t) * beta_t
#         x <- mu + torch::torch_sqrt(var) * torch::torch_randn_like(x)
#       } else {
#         x <- mu
#       }
#     }
#     return(as.matrix(x))
#   } else if (fit$type == "ddpm_cnn"){
#     stopifnot(!is.null(shape_img))
#     B <- n; H <- shape_img[1]; W <- shape_img[2]
#     x <- torch::torch_randn(B, 1, H, W)
#     for (ti in steps:1){
#       t_frac <- torch::torch_full(c(B,1), ti/fit$T)
#       eps <- fit$model(x, t_frac)
#       sa_t   <- torch::torch_tensor(array(sch$sqrt_alpha_bar[ti], dim = c(B,1,1,1)))
#       sa_tm1 <- torch::torch_tensor(array(ifelse(ti>1, sch$sqrt_alpha_bar[ti-1], 1), dim = c(B,1,1,1)))
#       om_t   <- torch::torch_tensor(array(1 - sch$alpha_bar[ti], dim = c(B,1,1,1)))
#       om_tm1 <- torch::torch_tensor(array(ifelse(ti>1, 1 - sch$alpha_bar[ti-1], 0), dim = c(B,1,1,1)))
#       beta_t <- torch::torch_tensor(array(sch$beta[ti], dim = c(B,1,1,1)))

#       x0_hat <- x / sa_t - torch::torch_sqrt(om_t) * eps
#       mu <- (sa_tm1 * beta_t / om_t) * x0_hat +
#             (torch::torch_sqrt(1-beta_t) * (om_tm1 / om_t)) * x
#       if (ti > 1){
#         var <- (om_tm1 / om_t) * beta_t
#         x <- mu + torch::torch_sqrt(var) * torch::torch_randn_like(x)
#       } else {
#         x <- mu
#       }
#     }
#     img <- (x$clamp(-1,1) + 1)/2
#     return(img$to(device="cpu")$squeeze(2)$numpy())
#   } else stop("Unknown fit$type")
# }
sample_ddpm <- function(fit, n = 64, steps = NULL, shape_img = NULL, seed = 123){
  set.seed(seed); torch::torch_manual_seed(seed)
  if(is.null(steps)) steps <- fit$T
  sch <- fit$schedule

  if (fit$type == "ddpm_mlp") {
    d <- fit$dim
    x <- torch::torch_randn(n, d)
    T <- fit$T

    for (ti in steps:1) {
      t_frac <- torch::torch_full(c(n, 1), ti / T)
      eps <- fit$model(x, t_frac)   # 预测噪声 epsilon_theta(x_t, t)

      beta_t      <- sch$beta[ti]
      alpha_t     <- sch$alpha[ti]
      alpha_bar_t <- sch$alpha_bar[ti]

      if (ti > 1) {
        alpha_bar_prev <- sch$alpha_bar[ti - 1]
      } else {
        alpha_bar_prev <- 1
      }

      # posterior variance β̃_t
      beta_t_tilde <- (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t

      # μ_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps)
      mu <- (1 / sqrt(alpha_t)) * (
        x - (beta_t / sqrt(1 - alpha_bar_t)) * eps
      )

      if (ti > 1) {
        x <- mu + sqrt(beta_t_tilde) * torch::torch_randn_like(x)
      } else {
        x <- mu
      }
    }

    return(as.matrix(x))
  }

  # # CNN 分支先暂时沿用你原来的，等 dist 这边跑通了再一起改
  # if (fit$type == "ddpm_cnn") {
  #   stopifnot(!is.null(shape_img))
  #   B <- n; H <- shape_img[1]; W <- shape_img[2]
  #   x <- torch::torch_randn(B, 1, H, W)
  #   T <- fit$T

  #   for (ti in steps:1) {
  #     t_frac <- torch::torch_full(c(B,1), ti / T)
  #     eps <- fit$model(x, t_frac)

  #     beta_t      <- sch$beta[ti]
  #     alpha_t     <- sch$alpha[ti]
  #     alpha_bar_t <- sch$alpha_bar[ti]
  #     if (ti > 1) {
  #       alpha_bar_prev <- sch$alpha_bar[ti - 1]
  #     } else {
  #       alpha_bar_prev <- 1
  #     }
  #     beta_t_tilde <- (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t

  #     mu <- (1 / sqrt(alpha_t)) * (
  #       x - (beta_t / sqrt(1 - alpha_bar_t)) * eps
  #     )

  #     if (ti > 1) {
  #       x <- mu + sqrt(beta_t_tilde) * torch::torch_randn_like(x)
  #     } else {
  #       x <- mu
  #     }
  #   }
  #   img <- (x$clamp(-1,1) + 1)/2
  #   return(img$to(device="cpu")$squeeze(2)$numpy())
  # }

  if (fit$type == "ddpm_cnn") {
    stopifnot(!is.null(shape_img))
    B <- n; H <- shape_img[1]; W <- shape_img[2]
    x <- torch::torch_randn(B, 1, H, W)
    T <- fit$T

    for (ti in steps:1) {
      t_frac <- torch::torch_full(c(B,1), ti / T)
      eps <- fit$model(x, t_frac)

      beta_t      <- sch$beta[ti]
      alpha_t     <- sch$alpha[ti]
      alpha_bar_t <- sch$alpha_bar[ti]
      if (ti > 1) {
        alpha_bar_prev <- sch$alpha_bar[ti - 1]
      } else {
        alpha_bar_prev <- 1
      }
      beta_t_tilde <- (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t

      mu <- (1 / sqrt(alpha_t)) * (
        x - (beta_t / sqrt(1 - alpha_bar_t)) * eps
      )

      if (ti > 1) {
        x <- mu + sqrt(beta_t_tilde) * torch::torch_randn_like(x)
      } else {
        x <- mu
      }
    }

    img <- (x$clamp(-1,1) + 1)/2  # 仍然是 tensor [B,1,H,W]

    # ✅ 正确写法：先搬到 CPU，squeeze 掉通道维，然后用 as.array 取出 R array
    img <- img$to(device = "cpu")$squeeze(2)
    return(as.array(img))  # [B,H,W]
  }

  stop("Unknown fit$type")
}


# Score from epsilon for VP SDE: score = -eps / sigma(t), sigma^2 = 1 - alpha_bar
score_from_eps <- function(eps_pred, sigma){ -eps_pred / sigma }

#' Reverse VP SDE sampling with Euler-Maruyama (optional alternative to DDPM)
#' @export
sample_vp_sde <- function(fit, n = 64, steps = 1000, shape_img = NULL, seed = 123){
  set.seed(seed); torch::torch_manual_seed(seed)
  sch <- fit$schedule
  if(fit$type == "ddpm_mlp"){
    d <- fit$dim
    x <- torch::torch_randn(n, d)
    dt <- 1.0/steps
    for (k in seq_len(steps)){
      t_idx <- max(1, floor((1 - (k-1)/steps) * fit$T))
      t_frac <- torch::torch_full(c(n,1), t_idx/fit$T)
      eps <- fit$model(x, t_frac)
      sigma <- torch::torch_tensor(matrix(sch$sqrt_one_minus_alpha_bar[t_idx], n, 1))
      score <- score_from_eps(eps, sigma)
      # call C++ step
      x <- x + 0 # no-op to ensure tensor; convert to R matrix for C++
      x_mat <- as.matrix(x)
      score_mat <- as.matrix(score)
      x_new <- vp_reverse_sde_step(x_mat, score_mat, beta_t = sch$beta[t_idx], dt = dt)
      x <- torch::torch_tensor(x_new)
    }
    return(as.matrix(x))
  } else if (fit$type == "ddpm_cnn"){
    stopifnot(!is.null(shape_img))
    B <- n; H <- shape_img[1]; W <- shape_img[2]
    x <- torch::torch_randn(B, 1, H, W)
    dt <- 1.0/steps
    for (k in seq_len(steps)){
      t_idx <- max(1, floor((1 - (k-1)/steps) * fit$T))
      t_frac <- torch::torch_full(c(B,1), t_idx/fit$T)
      eps <- fit$model(x, t_frac)
      sigma <- torch::torch_tensor(array(sch$sqrt_one_minus_alpha_bar[t_idx], dim = c(B,1,1,1)))
      score <- score_from_eps(eps, sigma)
      x_mat <- as.matrix(x$squeeze(2))
      score_mat <- as.matrix(score$squeeze(2))
      x_new <- vp_reverse_sde_step(x_mat, score_mat, beta_t = sch$beta[t_idx], dt = dt)
      x <- torch::torch_tensor(x_new)$unsqueeze(2)
    }
    img <- (x$clamp(-1,1) + 1)/2
    # return(img$to(device="cpu")$squeeze(2)$numpy())
    img <- img$to(device = "cpu")$squeeze(2)
    return(as.array(img))
  } else stop("Unknown fit$type")
}

# Simple denoise (single step toward x0_hat) for image
#' @export
denoise_ddpm <- function(fit, noisy_img, t_idx){
  stopifnot(fit$type == "ddpm_cnn")
  x <- torch::torch_tensor(noisy_img)$unsqueeze(1) # [B,1,H,W]
  B <- x$size()[1]
  sch <- fit$schedule
  t_frac <- torch::torch_full(c(B,1), t_idx/fit$T)
  eps <- fit$model(x, t_frac)
  sa_t <- torch::torch_tensor(array(sch$sqrt_alpha_bar[t_idx], dim = c(B,1,1,1)))
  om_t <- torch::torch_tensor(array(1 - sch$alpha_bar[t_idx], dim = c(B,1,1,1)))
  x0_hat <- x / sa_t - torch::torch_sqrt(om_t) * eps
  img <- (x0_hat$clamp(-1,1) + 1)/2
  # img$to(device="cpu")$squeeze(2)$numpy()
  img <- img$to(device = "cpu")$squeeze(2)
  as.array(img)
}

# Inpainting via DDPM: keep known pixels, diffuse+reverse on masked
#' @export
inpaint_ddpm <- function(fit, img, mask, steps = NULL, seed = 123){
  stopifnot(fit$type == "ddpm_cnn")
  set.seed(seed); torch::torch_manual_seed(seed)
  if(is.null(steps)) steps <- fit$T
  sch <- fit$schedule
  x0 <- torch::torch_tensor(img)$unsqueeze(1) * 2 - 1
  m  <- torch::torch_tensor(mask)$unsqueeze(1) # 1 for known pixels
  B <- x0$size()[1]; H <- x0$size()[3]; W <- x0$size()[4]
  # add noise to unknown region heavily
  x <- x0 * m + torch::torch_randn_like(x0) * (1 - m)
  for (ti in steps:1){
    t_frac <- torch::torch_full(c(B,1), ti/fit$T)
    eps <- fit$model(x, t_frac)
    sa_t   <- torch::torch_tensor(array(sch$sqrt_alpha_bar[ti], dim = c(B,1,1,1)))
    sa_tm1 <- torch::torch_tensor(array(ifelse(ti>1, sch$sqrt_alpha_bar[ti-1], 1), dim = c(B,1,1,1)))
    om_t   <- torch::torch_tensor(array(1 - sch$alpha_bar[ti], dim = c(B,1,1,1)))
    om_tm1 <- torch::torch_tensor(array(ifelse(ti>1, 1 - sch$alpha_bar[ti-1], 0), dim = c(B,1,1,1)))
    beta_t <- torch::torch_tensor(array(sch$beta[ti], dim = c(B,1,1,1)))

    x0_hat <- x / sa_t - torch::torch_sqrt(om_t) * eps
    mu <- (sa_tm1 * beta_t / om_t) * x0_hat +
          (torch::torch_sqrt(1-beta_t) * (om_tm1 / om_t)) * x
    if (ti > 1){
      var <- (om_tm1 / om_t) * beta_t
      x <- mu + torch::torch_sqrt(var) * torch::torch_randn_like(x)
    } else {
      x <- mu
    }
    # re-impose known pixels
    x <- x * (1 - m) + x0 * m
  }
  img <- (x$clamp(-1,1) + 1)/2
  # img$to(device="cpu")$squeeze(2)$numpy()
  img <- img$to(device = "cpu")$squeeze(2)
  as.array(img)
}
