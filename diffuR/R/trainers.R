# Loss is MSE between predicted noise and true noise
loss_ddpm_mse <- function(eps_pred, eps_true) {
  torch::nnf_mse_loss(eps_pred, eps_true, reduction = "mean")
}

# Distribution trainer (MLP)
#' @export
train_diffusion_dist <- function(X, epochs = 50, T = 1000, lr = 1e-3, batch_size = 256,
                                 schedule = beta_linear(T), verbose = TRUE, seed = 42){
  set.seed(seed)
  torch::torch_manual_seed(seed)
  X <- as.matrix(X); n <- nrow(X); d <- ncol(X)
  model <- build_mlp_eps(d)()
  opt <- torch::optim_adam(model$parameters, lr = lr)
  ds <- torch::tensor_dataset(torch::torch_tensor(X))
  dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = TRUE)

  for (e in seq_len(epochs)){
    model$train()
    total <- 0; nstep <- 0
    coro::loop(for (b in dl) {
      x0 <- b[[1]]
      bs <- x0$size()[1]
      t_idx <- sample.int(T, bs, replace = TRUE)
      t_frac <- torch::torch_tensor(matrix(t_idx / T, bs, 1))
      eps_true <- torch::torch_randn_like(x0)
      sa <- torch::torch_tensor(matrix(schedule$sqrt_alpha_bar[t_idx], bs, 1))
      som <- torch::torch_tensor(matrix(schedule$sqrt_one_minus_alpha_bar[t_idx], bs, 1))
      x_t <- sa * x0 + som * eps_true

      opt$zero_grad()
      eps_pred <- model(x_t, t_frac)
      loss <- loss_ddpm_mse(eps_pred, eps_true)
      loss$backward(); opt$step()
      total <- total + as.numeric(loss$item()); nstep <- nstep + 1
    })
    if(verbose) cat(sprintf("Epoch %d | loss=%.4f\n", e, total / max(nstep,1)))
  }
  list(model = model, schedule = schedule, T = T, type = "ddpm_mlp", dim = d)
}

# # Image trainer (CNN) for 28x28 grayscale (e.g., MNIST)
# #' @export
# train_diffusion_image <- function(train_dl, epochs = 3, T = 1000, lr = 2e-4,
#                                   schedule = beta_linear(T), verbose = TRUE, seed = 42){
#   set.seed(seed); torch::torch_manual_seed(seed)
#   model <- build_cnn_eps()()
#   opt <- torch::optim_adam(model$parameters, lr = lr)

#   for (e in seq_len(epochs)){
#     model$train(); total <- 0; nstep <- 0
#     coro::loop(for (b in train_dl) {
#       # # b: list(image, label) where image in [0,1], shape [B,1,28,28]
#       # x0 <- b[[1]]$to(dtype = torch::torch_float()) * 2 - 1  # scale to [-1,1]
#       # bs <- x0$size()[1]

#       # b[[1]]: [B,28,28] in [0,1]
#       img <- b[[1]]$to(dtype = torch::torch_float())

#       # 显式加一个 channel 维度: [B,28,28] -> [B,1,28,28]
#       if (img$ndim() == 3) {
#         x0 <- img$unsqueeze(2L)
#       } else {
#         x0 <- img
#       }

#       x0 <- x0 * 2 - 1  # [-1,1]
#       bs <- x0$size()[1]

#       t_idx <- sample.int(T, bs, replace = TRUE)
#       # t_frac 必须是向量形式,而不是矩阵
#       t_frac <- torch::torch_tensor(as.numeric(t_idx) / T, dtype = torch::torch_float())
#       eps_true <- torch::torch_randn_like(x0)
#       sa <- torch::torch_tensor(array(schedule$sqrt_alpha_bar[t_idx], dim = c(bs,1,1,1)))
#       som <- torch::torch_tensor(array(schedule$sqrt_one_minus_alpha_bar[t_idx], dim = c(bs,1,1,1)))
#       x_t <- sa * x0 + som * eps_true

#       opt$zero_grad()
#       eps_pred <- model(x_t, t_frac)
#       loss <- loss_ddpm_mse(eps_pred, eps_true)
#       loss$backward(); opt$step()
#       total <- total + as.numeric(loss$item()); nstep <- nstep + 1
#     })
#     if(verbose) cat(sprintf("Epoch %d | loss=%.4f\n", e, total / max(nstep,1)))
#   }
#   list(model = model, schedule = schedule, T = T, type = "ddpm_cnn")
# }


# # Image trainer (CNN) for 28x28 grayscale (e.g., MNIST)
# #' @export
# train_diffusion_image <- function(train_dl, epochs = 3, T = 1000, lr = 2e-4,
#                                   schedule = beta_linear(T), verbose = TRUE, seed = 42){
#   set.seed(seed); torch::torch_manual_seed(seed)
#   model <- build_cnn_eps()()
#   opt <- torch::optim_adam(model$parameters, lr = lr)

#   for (e in seq_len(epochs)){
#     model$train(); total <- 0; nstep <- 0

#     coro::loop(for (b in train_dl) {

#       # b[[1]]: 期望形状 [B, 28, 28] in [0,1]
#       img <- b[[1]]$to(dtype = torch::torch_float())

#       # 明确检查维度数
#       nd <- length(img$size())

#       if (nd == 3) {
#         # [B,28,28] -> [B,1,28,28]
#         x0 <- img$unsqueeze(2L)
#       } else if (nd == 4) {
#         # 已经是 [B,1,28,28]
#         x0 <- img
#       } else {
#         stop("Unexpected image tensor shape: ", paste(img$size(), collapse = " x "))
#       }

#       # scale to [-1,1]
#       x0 <- x0 * 2 - 1
#       bs <- x0$size()[1]

#       # 采样时间步 t
#       t_idx <- sample.int(T, bs, replace = TRUE)
#       # t_frac 用一维向量，cnn 里会处理成 [B,1]
#       t_frac <- torch::torch_tensor(
#         as.numeric(t_idx) / T,
#         dtype = torch::torch_float()
#       )

#       eps_true <- torch::torch_randn_like(x0)

#       sa  <- torch::torch_tensor(array(schedule$sqrt_alpha_bar[t_idx],
#                                        dim = c(bs,1,1,1)))
#       som <- torch::torch_tensor(array(schedule$sqrt_one_minus_alpha_bar[t_idx],
#                                        dim = c(bs,1,1,1)))

#       x_t <- sa * x0 + som * eps_true

#       opt$zero_grad()
#       eps_pred <- model(x_t, t_frac)
#       loss <- loss_ddpm_mse(eps_pred, eps_true)
#       loss$backward()
#       opt$step()

#       total <- total + as.numeric(loss$item())
#       nstep <- nstep + 1
#     })

#     if (verbose) {
#       cat(sprintf("Epoch %d | loss=%.4f\n", e, total / max(nstep, 1)))
#     }
#   }

#   list(model = model, schedule = schedule, T = T, type = "ddpm_cnn")
# }

# Image trainer (UNet) for 28x28 grayscale (e.g., MNIST)
#' @export
train_diffusion_image <- function(train_dl, epochs = 3, T = 1000, lr = 1e-4,
                                  schedule = beta_linear(T), verbose = TRUE, seed = 42,
                                  use_unet = TRUE, model_arch = NULL) {
  set.seed(seed)
  torch::torch_manual_seed(seed)
  
  # 选择模型架构
  if (use_unet) {
    # UNet：参数数量多但效果好
    model <- build_unet_eps(ch = 64, ch_mult = c(1, 2), t_dim = 128, num_res_blocks = 2)()
    cat("Using UNet architecture (recommended for better quality)\n")
  } else if (!is.null(model_arch)) {
    # 自定义模型
    model <- model_arch()
  } else {
    # 回退到简单CNN
    model <- build_cnn_eps()()
    cat("Using simple CNN (baseline)\n")
  }
  
  # 使用梯度剪裁防止爆炸
  opt <- torch::optim_adam(model$parameters, lr = lr)
  
  for (e in seq_len(epochs)) {
    model$train()
    total <- 0
    nstep <- 0
    max_grad_norm <- 1.0  # 梯度剪裁阈值

    coro::loop(for (b in train_dl) {
      # b[[1]]: [28, 28, B] in [0,1]
      img <- b[[1]]$to(dtype = torch::torch_float())

      # 维度重排: [H,W,B] -> [B,H,W] -> [B,1,H,W]
      x0 <- img$permute(c(3L, 1L, 2L))$unsqueeze(2L)

      # scale to [-1,1]
      x0 <- x0 * 2 - 1
      bs <- x0$size()[1]

      # 采样时间步 t
      t_idx <- sample.int(T, bs, replace = TRUE)
      t_frac <- torch::torch_tensor(
        as.numeric(t_idx) / T,
        dtype = torch::torch_float()
      )

      eps_true <- torch::torch_randn_like(x0)

      sa <- torch::torch_tensor(
        array(schedule$sqrt_alpha_bar[t_idx], dim = c(bs, 1, 1, 1))
      )
      som <- torch::torch_tensor(
        array(schedule$sqrt_one_minus_alpha_bar[t_idx], dim = c(bs, 1, 1, 1))
      )

      x_t <- sa * x0 + som * eps_true

      opt$zero_grad()
      eps_pred <- model(x_t, t_frac)
      loss <- loss_ddpm_mse(eps_pred, eps_true)
      loss$backward()
      
      # 梯度剪裁
      torch::nn_utils_clip_grad_norm_(model$parameters, max_grad_norm)
      
      opt$step()

      total <- total + as.numeric(loss$item())
      nstep <- nstep + 1
    })

    if (verbose) {
      avg_loss <- total / max(nstep, 1)
      cat(sprintf("Epoch %d | loss=%.4f\n", e, avg_loss))
    }
  }

  list(model = model, schedule = schedule, T = T, type = "ddpm_cnn")
  # list(model = model, schedule = schedule, T = T, type = "ddpm_unet")
}
