# ========================================
# MNIST Diffusion 优化示例
# 对比：简单CNN vs UNet
# ========================================

library(torch)
library(torchvision)
library(coro)
library(ggplot2)

# 加载模型和训练函数
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/model_cnn.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/model_mlp.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/schedules.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/trainers.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/sampling.R")

# ========== 步骤 1: 准备数据 ==========
cat("Loading MNIST data...\n")
train_dl <- mnist_train_dataloader(batch_size = 256)

# ========== 步骤 2: 定义调度 ==========
# Cosine schedule 比线性schedule更稳定
schedule_cosine <- beta_cosine(T = 1000)
schedule_linear <- beta_linear(T = 1000)

# ========== 步骤 3: 快速验证 (5 epochs) ==========
cat("\n========== QUICK VALIDATION (5 epochs) ==========\n")
cat("Training UNet (5 epochs)...\n")

fit_unet_quick <- train_diffusion_image(
  train_dl = train_dl,
  epochs   = 5,
  T        = 1000,
  lr       = 1e-4,
  schedule = schedule_cosine,
  verbose  = TRUE,
  seed     = 42,
  use_unet = TRUE
)

cat("\nTraining simple CNN (5 epochs)...\n")
fit_cnn_quick <- train_diffusion_image(
  train_dl = train_dl,
  epochs   = 5,
  T        = 1000,
  lr       = 5e-4,
  schedule = schedule_linear,
  verbose  = TRUE,
  seed     = 42,
  use_unet = FALSE
)

# ========== 步骤 4: 完整训练 (推荐 100 epochs) ==========
# 取消下面注释以运行完整训练（约 30-60分钟）

# cat("\n========== FULL TRAINING (100 epochs) ==========\n")
# cat("Training UNet (100 epochs)...\n")
# 
# fit_unet <- train_diffusion_image(
#   train_dl = train_dl,
#   epochs   = 100,
#   T        = 1000,
#   lr       = 1e-4,
#   schedule = schedule_cosine,
#   verbose  = TRUE,
#   seed     = 42,
#   use_unet = TRUE
# )

# ========== 步骤 5: 采样和可视化 ==========
cat("\n========== SAMPLING ==========\n")

# 从快速验证的模型采样（质量一般，但快速）
samples_unet <- sample_ddpm(
  fit       = fit_unet_quick,
  n         = 16L,
  steps     = fit_unet_quick$T,
  shape_img = c(28, 28)
)

samples_cnn <- sample_ddpm(
  fit       = fit_cnn_quick,
  n         = 16L,
  steps     = fit_cnn_quick$T,
  shape_img = c(28, 28)
)

cat("Samples generated. Shapes:", dim(samples_unet), "\n")

# ========== 步骤 6: 可视化结果 ==========
cat("Plotting results...\n")

# 函数：绘制16张图像
plot_samples <- function(samples, title = "Generated Samples") {
  par(mfrow = c(4, 4), mar = c(0.1, 0.1, 0.1, 0.1))
  for (i in 1:16) {
    img <- samples[i, , ]
    image(1:28, 1:28, t(apply(img, 2, rev)),
          col = gray.colors(256),
          axes = FALSE,
          main = "")
  }
  mtext(title, side = 3, outer = TRUE, cex = 1.5)
}

# 绘制UNet样本
plot_samples(samples_unet, "UNet Generated Samples (5 epochs)")

# 绘制CNN样本
plot_samples(samples_cnn, "Simple CNN Generated Samples (5 epochs)")

# ========== 步骤 7: 分析和对比 ==========
cat("\n========== ANALYSIS ==========\n")
cat("UNet samples range:", range(samples_unet), "\n")
cat("CNN samples range:", range(samples_cnn), "\n")

cat("\nExpected improvements with UNet (after 100 epochs):\n")
cat("- Clearer digit boundaries\n")
cat("- More coherent strokes\n")
cat("- Better diversity\n")
cat("- Fewer blurry artifacts\n")

# ========== 步骤 8: 保存模型（可选） ==========
# torch::torch_save(fit_unet$model, "mnist_unet_model.pt")
# torch::torch_save(fit_cnn_quick$model, "mnist_cnn_model.pt")

cat("\n========== OPTIMIZATION TIPS ==========\n")
cat("1. Use UNet for better quality (recommended)\n")
cat("2. Use beta_cosine schedule instead of beta_linear\n")
cat("3. Lower learning rate for UNet: 1e-4 instead of 5e-4\n")
cat("4. Increase batch size to 256 for stability\n")
cat("5. Train for 100+ epochs for best results\n")
cat("6. Monitor loss curves to prevent overfitting\n")
cat("\nFor more details, see: UNET_OPTIMIZATION_GUIDE.md\n")
