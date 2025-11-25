#!/usr/bin/env Rscript
# ============================================================================
# Improved Diffusion Model Training & Evaluation Pipeline
# 演示：train/test/val 划分 + 定量评估指标
# ============================================================================

# 源代码
library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/src/kernels.cpp")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/datasets.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/metrics.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/model_cnn.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/model_mlp.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/parallel.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/sampling.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/schedules.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/trainers.R")

library(torch)
library(ggplot2)
library(coro)

# ============================================================================
# 示例 1: MLP 模型 - Swiss Roll 数据
# ============================================================================
cat("\n")
cat(strrep("=", 79), "\n")
cat("EXAMPLE 1: MLP Diffusion on Swiss Roll with Train/Val/Test Split\n")
cat(strrep("=", 79), "\n")


# 1. 生成完整数据集
X_full <- swiss_roll(n = 2000, noise = 0.1, seed = 1)

# 2. 标准化（可逆）
Xmean <- colMeans(X_full)
Xsd   <- apply(X_full, 2, sd)
X_full_std <- scale(X_full, center = Xmean, scale = Xsd)

# 3. 数据划分：70% train, 15% val, 15% test
set.seed(42)
splits <- train_test_split(X_full_std, train_ratio = 0.7, val_ratio = 0.15, seed = 42)

cat(sprintf("\n✓ Data split: train=%d, val=%d, test=%d\n", 
            nrow(splits$train), nrow(splits$val), nrow(splits$test)))

# 4. 定义 schedule
sch <- beta_cosine(T = 1000)

# 5. 训练模型（仅在训练集上）
cat("\n--- Training MLP Diffusion Model ---\n")
fit_mlp <- train_diffusion_dist(
  X         = splits$train,  # ✅ 只在训练集上训练
  epochs    = 200,
  T         = 1000,
  lr        = 1e-4,
  batch_size = 256,
  schedule  = sch,
  verbose   = TRUE,
  seed      = 42
)

# 6. 在验证集上采样评估
cat("\n--- Sampling & Evaluating on Validation Set ---\n")
n_val_samples <- nrow(splits$val)
fake_val_std <- sample_ddpm(
  fit   = fit_mlp,
  n     = n_val_samples,
  steps = fit_mlp$T,
  seed  = 123
)

# 7. 评估指标
cat("\n--- Validation Set Metrics ---\n")
metrics_val <- evaluate_samples(
  X_real = splits$val,
  X_fake = fake_val_std,
  type   = "distribution",
  verbose = TRUE
)

# 8. 在测试集上最终评估
cat("\n--- Final Evaluation on Test Set ---\n")
n_test_samples <- nrow(splits$test)
fake_test_std <- sample_ddpm(
  fit   = fit_mlp,
  n     = n_test_samples,
  steps = fit_mlp$T,
  seed  = 456
)

metrics_test <- evaluate_samples(
  X_real = splits$test,
  X_fake = fake_test_std,
  type   = "distribution",
  verbose = TRUE
)

# 9. 可视化：比较训练集中原始数据与采样数据
fake_full_std <- sample_ddpm(
  fit   = fit_mlp,
  n     = nrow(splits$train),
  steps = fit_mlp$T,
  seed  = 789
)

# 还原到原坐标
fake_full <- sweep(fake_full_std, 2, Xsd, `*`)
fake_full <- sweep(fake_full, 2, Xmean, `+`)
colnames(fake_full) <- c("x1", "x2")


# 🔹 这里插入：按半径的 99% quantile 去除 outlier
fake_full_df <- as.data.frame(fake_full)

# 计算每个点到原点的距离
r <- sqrt(fake_full_df$x1^2 + fake_full_df$x2^2)

# 找到 99% 分位数阈值
r_thr <- quantile(r, 0.99)

# 只保留半径 <= 99% 分位数的点
fake_full_trim <- fake_full_df[r <= r_thr, ]

cat(sprintf("Removed %d outliers (top 1%% by radius)\n",
            nrow(fake_full_df) - nrow(fake_full_trim)))


X_train_orig <- sweep(splits$train, 2, Xsd, `*`)
X_train_orig <- sweep(X_train_orig, 2, Xmean, `+`)
colnames(X_train_orig) <- c("x1", "x2")
# 
# p1 <- ggplot(data.frame(X_train_orig, type = "real"), aes(x = x1, y = x2, color = type)) +
#   geom_point(alpha = 0.5, size = 1) +
#   coord_equal() +
#   scale_color_manual(values = c("real" = "#00BFC4")) +
#   theme_minimal(base_size = 12) +
#   labs(title = "Original Training Data", x = "x1", y = "x2")
# 
# p2 <- ggplot(data.frame(fake_full_trim, type = "fake"),
#              aes(x = x1, y = x2, color = type)) +
#   geom_point(alpha = 0.5, size = 1) +
#   coord_equal() +
#   scale_color_manual(values = c("fake" = "#F8766D")) +
#   theme_minimal(base_size = 12) +
#   labs(title = "Generated Samples (99% quantile trimmed)",
#        x = "x1", y = "x2")

# 先准备两个数据框，加上标记列
df_real <- data.frame(
  X_train_orig,
  type = "real",
  panel = "Training data"
)

df_fake <- data.frame(
  fake_full_trim,
  type = "fake",
  panel = "Generated (99% trimmed)"
)

# 合在一起
df_all <- rbind(df_real, df_fake)

# 画图：一行两个 facet
p_all <- ggplot(df_all, aes(x = x1, y = x2, color = type)) +
  geom_point(alpha = 0.45, size = 0.6) +
  coord_equal() +
  facet_wrap(~ panel, nrow = 1) +   # 或者 facet_grid(. ~ panel)
  scale_color_manual(values = c("real" = "#00BFC4",
                                "fake" = "#F8766D")) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "x1",
    y = "x2",
    color = "type",
    title = "Real vs Generated swiss roll (99% quantile trimmed)"
  )


# ============================================================================
# 示例 2: UNet 模型 - MNIST 数据
# ============================================================================
cat("\n")
cat("=" %*% rep("=", 79))
cat("\n")
cat("EXAMPLE 2: UNet Diffusion on MNIST with Train/Val/Test Split")
cat("\n")
cat("=" %*% rep("=", 79))
cat("\n")

# 1. 设置多线程
torch::torch_set_num_threads(8)
torch::torch_set_num_interop_threads(2)

# 2. 加载和划分 MNIST 数据
cat("\n--- Loading and Splitting MNIST Dataset ---\n")
mnist_splits <- mnist_split(
  train_ratio = 0.7,
  val_ratio   = 0.15,
  batch_size  = 512,
  num_workers = 2L,
  seed        = 42
)

cat("✓ MNIST splits loaded: train_dl, val_dl, test_dl\n")

# 3. 定义 schedule
schedule_cosine <- beta_cosine(T = 1000)

# 4. 训练 UNet 模型（仅在训练集上）
cat("\n--- Training UNet Diffusion Model ---\n")
t0_train <- Sys.time()
fit_unet <- train_diffusion_image(
  train_dl  = mnist_splits$train_dl,
  epochs    = 5,  # 演示用，可增加
  T         = 1000,
  lr        = 1e-4,
  schedule  = schedule_cosine,
  verbose   = TRUE,
  seed      = 42,
  use_unet  = TRUE
)
t1_train <- Sys.time()
cat(sprintf("✓ Training completed in %.2f minutes\n", 
            as.numeric(difftime(t1_train, t0_train, units = "mins"))))

# 5. 在验证集上采样（演示小规模采样）
cat("\n--- Sampling on Validation Set ---\n")
t0_val_sample <- Sys.time()
val_samples <- sample_ddpm(
  fit       = fit_unet,
  n         = 16,  # 采样 16 张测试
  steps     = 500,  # 用 500 步加速（完整是 T=1000）
  shape_img = c(28, 28),
  seed      = 123
)
t1_val_sample <- Sys.time()
cat(sprintf("✓ Sampled 16 images in %.2f seconds\n", 
            as.numeric(difftime(t1_val_sample, t0_val_sample, units = "secs"))))

# 6. 从验证集提取样本进行定量评估
cat("\n--- Extracting Real Samples from Validation Set ---\n")
# 从 val dataloader 获取部分真实数据
val_it <- mnist_splits$val_dl$.iter()
real_samples_list <- list()
for (i in 1:2) {
  b <- val_it$.next()
  img_batch <- b[[1]]$to(dtype = torch::torch_float())
  # img_batch shape: [batch_size, 28, 28]
  # 转换为 [batch_size, 28, 28] 数组
  real_samples_list[[i]] <- as.array(img_batch)
}
X_real_mnist <- abind::abind(real_samples_list[[1]], real_samples_list[[2]], along = 1)
dim(X_real_mnist)  # 应该是 [batch_size*2, 28, 28]

# 7. 评估指标
cat("\n--- Image Evaluation Metrics ---\n")
if (dim(X_real_mnist)[1] == dim(val_samples)[1]) {
  metrics_mnist <- evaluate_samples(
    X_real = X_real_mnist,
    X_fake = val_samples,
    type   = "image",
    verbose = TRUE
  )
}

# 8. 可视化生成的图像
cat("\n--- Visualizing Generated Images ---\n")
par(mfrow = c(4, 4), mar = c(0.1, 0.1, 0.1, 0.1))
for (i in 1:16) {
  img <- val_samples[i, , ]  # [28, 28]
  image(
    1:28, 1:28,
    t(apply(img, 2, rev)),
    col  = gray.colors(256),
    axes = FALSE
  )
}
par(mfrow = c(1, 1))

cat("\n✓ Visualization complete!\n")

# ============================================================================
# 示例 3: 前向扩散过程可视化
# ============================================================================
cat("\n")
cat("=" %*% rep("=", 79))
cat("\n")
cat("EXAMPLE 3: Forward Diffusion Process Visualization")
cat("\n")
cat("=" %*% rep("=", 79))
cat("\n")

# 生成 8-Gaussians 数据
X <- gauss8_ring(n = 1000, radius = 2, noise = 0.1, seed = 1)

Xmean <- colMeans(X)
Xsd   <- apply(X, 2, sd)
X_std <- scale(X, center = Xmean, scale = Xsd)

sch <- beta_cosine(T = 1000)

# 选择时间步
t_show <- c(0, 100, 250, 500, 1000)

# 构造各时间步下的加噪样本
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

# 裁剪半径并还原
r2_real <- rowSums(X_std^2)
R <- sqrt(quantile(r2_real, 0.99))

X_real_clip_std <- X_std[r2_real <= R^2, , drop = FALSE]
X_real_clip <- sweep(X_real_clip_std, 2, Xsd, `*`)
X_real_clip <- sweep(X_real_clip, 2, Xmean, `+`)
colnames(X_real_clip) <- c("x1", "x2")

# 构造 dataframe
df_list <- list()
for (nm in names(snap_std)) {
  Xt_std <- snap_std[[nm]]
  r2 <- rowSums(Xt_std^2)
  keep <- r2 <= R^2
  Xt_std_clip <- Xt_std[keep, , drop = FALSE]
  
  Xt <- sweep(Xt_std_clip, 2, Xsd, `*`)
  Xt <- sweep(Xt, 2, Xmean, `+`)
  colnames(Xt) <- c("x1", "x2")
  
  step_lab <- paste0("t = ", sub("t", "", nm))
  
  df_noised <- data.frame(
    x1 = Xt[, 1],
    x2 = Xt[, 2],
    type = "noised",
    step = step_lab
  )
  df_real <- data.frame(
    x1 = X_real_clip[, 1],
    x2 = X_real_clip[, 2],
    type = "real",
    step = step_lab
  )
  
  df_list[[length(df_list) + 1]] <- rbind(df_real, df_noised)
}

df_all <- do.call(rbind, df_list)
df_all$step <- factor(df_all$step, levels = paste0("t = ", t_show))

# 画图
p_forward <- ggplot(df_all, aes(x = x1, y = x2, color = type)) +
  geom_point(alpha = 0.45, size = 0.6) +
  coord_equal() +
  facet_grid(. ~ step) +
  scale_color_manual(values = c("real" = "#00BFC4", "noised" = "#F8766D")) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "x1", y = "x2", color = "type",
    title = "Forward Diffusion: 8-Gaussians Ring"
  )

print(p_forward)

# ============================================================================
# 总结
# ============================================================================
cat("\n")
cat("=" %*% rep("=", 79))
cat("\n")
cat("SUMMARY")
cat("\n")
cat("=" %*% rep("=", 79))
cat("\n")

cat("\n✓ Successfully demonstrated:")
cat("\n  1. Train/Val/Test data splitting (70/15/15)")
cat("\n  2. MLP diffusion model training on synthetic distributions")
cat("\n  3. UNet diffusion model training on MNIST")
cat("\n  4. Distribution matching metrics:")
cat("\n     - MMD (Maximum Mean Discrepancy)")
cat("\n     - Frechet Distance")
cat("\n     - Wasserstein Distance")
cat("\n     - Coverage & Density")
cat("\n  5. Forward diffusion process visualization")
cat("\n")
cat("✓ All pipeline components integrated successfully!")
cat("\n")
