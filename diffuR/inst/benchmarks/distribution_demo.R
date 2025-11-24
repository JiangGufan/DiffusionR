# #!/usr/bin/env Rscript
# # Distribution Demo Script for diffuR package
# # Trains a diffusion model on Swiss Roll dataset
# # Usage: Rscript inst/benchmarks/distribution_demo.R

# # ============================================================================
# # PART 1: Load dependencies
# # ============================================================================
# cat(paste0(rep("=", 80), collapse = ""), "\n")
# cat("Distribution Diffusion Demo - Swiss Roll\n")
# cat(paste0(rep("=", 80), collapse = ""), "\n\n")

# library(diffuR)

# # Check for ggplot2
# if (!requireNamespace("ggplot2", quietly = TRUE)) {
#   cat("WARNING: ggplot2 not found. Installing...\n")
#   install.packages("ggplot2")
# }

# cat("✓ Dependencies loaded\n\n")

# # ============================================================================
# # PART 2: Load data
# # ============================================================================
# cat("Creating Swiss Roll dataset...\n")
# cat(paste0(rep("-", 80), collapse = ""), "\n")

# set.seed(1)
# X <- swiss_roll(n = 4000, noise = 0.15, seed = 1)

# cat("✓ Swiss Roll created successfully\n")
# cat("  - Number of samples:", nrow(X), "\n")
# cat("  - Dimension:", ncol(X), "\n")
# cat("  - X1 range: [", min(X[,1]), ", ", max(X[,1]), "]\n", sep = "")
# cat("  - X2 range: [", min(X[,2]), ", ", max(X[,2]), "]\n", sep = "")
# cat("\n")

# # ============================================================================
# # PART 3: Create noise schedule
# # ============================================================================
# cat("Creating noise schedule...\n")
# cat(paste0(rep("-", 80), collapse = ""), "\n")

# schedule <- beta_linear(T = 500, beta_min = 1e-4, beta_max = 0.02)

# cat("✓ Noise schedule created\n")
# cat("  - Type: Linear\n")
# cat("  - Timesteps: 500\n")
# cat("  - Beta range: [", min(schedule$beta), ", ", max(schedule$beta), "]\n", sep = "")
# cat("\n")

# # ============================================================================
# # PART 4: Train model
# # ============================================================================
# cat("Training diffusion model on Swiss Roll...\n")
# cat(paste0(rep("-", 80), collapse = ""), "\n")

# set.seed(1)
# torch::torch_manual_seed(1)

# fit <- train_diffusion_dist(
#   X = X,
#   epochs = 15,
#   T = 500,
#   lr = 1e-3,
#   batch_size = 1024,
#   schedule = schedule,
#   verbose = TRUE,
#   seed = 1
# )

# cat("\n✓ Training completed successfully\n")
# cat("  - Model type:", fit$type, "\n")
# cat("  - Data dimension:", fit$dim, "\n")
# cat("  - Timesteps:", fit$T, "\n")
# cat("\n")

# # ============================================================================
# # PART 5: Generate samples
# # ============================================================================
# cat("Generating samples from trained model...\n")
# cat(paste0(rep("-", 80), collapse = ""), "\n")

# samp <- sample_ddpm(fit, n = 4000, steps = 500, seed = 123)
# colnames(samp) <- colnames(X)

# cat("✓ Sampling completed successfully\n")
# cat("  - Number of samples:", nrow(samp), "\n")
# cat("  - X1 range: [", min(samp[,1]), ", ", max(samp[,1]), "]\n", sep = "")
# cat("  - X2 range: [", min(samp[,2]), ", ", max(samp[,2]), "]\n", sep = "")
# cat("\n")

# # ============================================================================
# # PART 6: Create visualization
# # ============================================================================
# cat("Creating comparison plot...\n")
# cat(paste0(rep("-", 80), collapse = ""), "\n")

# tryCatch({
#   p <- plot_2d_samples(X, samp)
  
#   output_file <- "dist_demo.png"
#   ggplot2::ggsave(
#     filename = output_file,
#     plot = p,
#     width = 10,
#     height = 5,
#     dpi = 150
#   )
  
#   cat("✓ Visualization saved to:", output_file, "\n")
#   cat("  - Left: Original Swiss Roll data (", nrow(X), " samples)\n", sep = "")
#   cat("  - Right: Generated samples from diffusion model (", nrow(samp), " samples)\n", sep = "")
# }, error = function(e) {
#   cat("✗ Error creating visualization:\n")
#   cat("  ", e$message, "\n")
# })

# cat("\n")
# cat(paste0(rep("=", 80), collapse = ""), "\n")
# cat("Distribution demo completed!\n")
# cat(paste0(rep("=", 80), collapse = ""), "\n")


## inst/benchmarks/distribution_demo.R
##
## 演示：在多种 2D toy 分布上，训练 diffusion 模型并可视化 forward diffusion
## 依赖：swiss_roll(), gauss8_ring(), heart2d(),
##       beta_cosine(), train_diffusion_dist(), sample_ddpm(), q_sample_xt_given_x0()

library(diffuR)
library(torch)
library(ggplot2)

# ------------- 1. 统一的 demo 函数 ------------- #

run_diffusion_toy_demo <- function(
  toy = c("swiss", "gauss8", "heart"),
  n = 2000,
  noise = 0.1,
  epochs = 200,
  T = 1000,
  seed = 42
) {
  toy <- match.arg(toy)

  set.seed(seed)
  torch::torch_manual_seed(seed)

  # ---- 1. 生成数据 ----
  X <- switch(
    toy,
    "swiss"  = swiss_roll(n = n, noise = noise, seed = seed),
    "gauss8" = gauss8_ring(n = n, radius = 2, noise = noise, seed = seed),
    "heart"  = heart2d(n = n, noise = noise, seed = seed)
  )

  # ---- 2. 标准化（可逆）----
  Xmean <- colMeans(X)
  Xsd   <- apply(X, 2, sd)
  X_std <- scale(X, center = Xmean, scale = Xsd)

  # ---- 3. 噪声 schedule + 训练 ----
  sch <- beta_cosine(T = T)

  fit_dist <- train_diffusion_dist(
    X          = X_std,
    epochs     = epochs,
    T          = T,
    lr         = 1e-4,
    batch_size = 512,
    schedule   = sch,
    verbose    = TRUE,
    seed       = seed
  )

  # ---- 4. forward diffusion 可视化 ----
  t_show <- c(0, 100, 200, 500, 1000)

  snap_std <- list()
  snap_std[["t0"]] <- as.matrix(X_std)

  set.seed(123)
  for (t in t_show[-1]) {
    snap_std[[paste0("t", t)]] <- q_sample_xt_given_x0(
      X0 = as.matrix(X_std),
      sa = sch$sqrt_alpha_bar[t],
      om = sch$sqrt_one_minus_alpha_bar[t]
    )
  }

  # ---- 5. 统一半径裁剪（防 outlier）----
  r2_real <- rowSums(X_std^2)
  R <- sqrt(quantile(r2_real, 0.99))

  X_real_clip_std <- X_std[r2_real <= R^2, , drop = FALSE]
  X_real_clip <- sweep(X_real_clip_std, 2, Xsd,   `*`)
  X_real_clip <- sweep(X_real_clip,     2, Xmean, `+`)
  colnames(X_real_clip) <- c("x1", "x2")

  # ---- 6. 组装成 df_all，用 facet_grid 横排 ----
  df_list <- list()

  for (nm in names(snap_std)) {
    Xt_std <- snap_std[[nm]]

    r2   <- rowSums(Xt_std^2)
    keep <- r2 <= R^2
    Xt_std_clip <- Xt_std[keep, , drop = FALSE]

    Xt <- sweep(Xt_std_clip, 2, Xsd,   `*`)
    Xt <- sweep(Xt,          2, Xmean, `+`)
    colnames(Xt) <- c("x1", "x2")

    step_lab <- paste0("t = ", sub("t", "", nm))

    df_fake <- data.frame(
      x1   = Xt[, 1],
      x2   = Xt[, 2],
      type = "noised",
      step = step_lab
    )
    df_real <- data.frame(
      x1   = X_real_clip[, 1],
      x2   = X_real_clip[, 2],
      type = "real",
      step = step_lab
    )

    df_list[[length(df_list) + 1]] <- rbind(df_real, df_fake)
  }

  df_all <- do.call(rbind, df_list)

  df_all$step <- factor(df_all$step,
                        levels = paste0("t = ", t_show))

  title_txt <- switch(
    toy,
    "swiss"  = "Forward diffusion of Swiss roll (cosine schedule)",
    "gauss8" = "Forward diffusion of 8-Gaussians ring (cosine schedule)",
    "heart"  = "Forward diffusion of Heart shape (cosine schedule)"
  )

  p <- ggplot(df_all, aes(x = x1, y = x2, color = type)) +
    geom_point(alpha = 0.45, size = 0.6) +
    coord_equal() +
    facet_grid(. ~ step) +
    scale_color_manual(values = c("real" = "#00BFC4", "noised" = "#F8766D")) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position  = "right",
      panel.grid.minor = element_blank()
    ) +
    labs(
      x = "x1",
      y = "x2",
      color = "type",
      title = title_txt
    )

  print(p)

  invisible(list(
    X        = X,
    X_std    = X_std,
    fit      = fit_dist,
    schedule = sch,
    plot     = p
  ))
}

# ------------- 2. 示例调用（用户可以注释/取消注释） ------------- #

# 示例 1：Swiss roll
# res_swiss <- run_diffusion_toy_demo(toy = "swiss")

# 示例 2：8-Gaussians ring
# res_gauss8 <- run_diffusion_toy_demo(toy = "gauss8")

# 示例 3：Heart
# res_heart <- run_diffusion_toy_demo(toy = "heart")
