# ============================================================================
# 改进版：完整的训练-验证-测试-评估流程
# ============================================================================
# 这部分代码演示如何使用新的数据划分和评估功能

# ===== 快速示例：MLP + Swiss Roll （包含数据划分和评估）=====
cat("\n" %*% rep("=", 80))
cat("\nMLP Diffusion Model with Proper Train/Val/Test Pipeline")
cat("\n" %*% rep("=", 80))

# 1. 生成数据
X_full <- swiss_roll(n = 2000, noise = 0.1, seed = 1)
Xmean <- colMeans(X_full)
Xsd   <- apply(X_full, 2, sd)
X_std <- scale(X_full, center = Xmean, scale = Xsd)

# 2. 数据划分（70% train, 15% val, 15% test）
set.seed(42)
data_splits <- train_test_split(X_std, train_ratio = 0.7, val_ratio = 0.15, seed = 42)

cat(sprintf("\nData split: train=%d, val=%d, test=%d\n", 
            nrow(data_splits$train), nrow(data_splits$val), nrow(data_splits$test)))

# 3. 定义 schedule
sch <- beta_cosine(T = 1000)

# 4. 在训练集上训练
cat("\nTraining on training set...\n")
fit_mlp <- train_diffusion_dist(
  X         = data_splits$train,  # ✅ 只在 train 上训练
  epochs    = 50,
  T         = 1000,
  lr        = 1e-4,
  batch_size = 256,
  schedule  = sch,
  verbose   = TRUE,
  seed      = 42
)

# 5. 在验证集上采样和评估
cat("\n\n--- Validation Set Evaluation ---\n")
fake_val <- sample_ddpm(
  fit   = fit_mlp,
  n     = nrow(data_splits$val),
  steps = 1000,
  seed  = 123
)

metrics_val <- evaluate_samples(
  X_real = data_splits$val,
  X_fake = fake_val,
  type   = "distribution",
  verbose = TRUE
)

# 6. 在测试集上最终评估
cat("\n--- Test Set Evaluation (Final) ---\n")
fake_test <- sample_ddpm(
  fit   = fit_mlp,
  n     = nrow(data_splits$test),
  steps = 1000,
  seed  = 456
)

metrics_test <- evaluate_samples(
  X_real = data_splits$test,
  X_fake = fake_test,
  type   = "distribution",
  verbose = TRUE
)

cat("\n✓ MLP Pipeline Complete!\n")

# ===== MNIST + UNet （包含数据划分和评估）=====
cat("\n" %*% rep("=", 80))
cat("\nUNet Diffusion Model with MNIST Train/Val/Test Pipeline")
cat("\n" %*% rep("=", 80))

# 1. 设置线程
torch::torch_set_num_threads(8)
torch::torch_set_num_interop_threads(2)

# 2. 加载并划分 MNIST
cat("\nLoading and splitting MNIST...\n")
mnist_data <- mnist_split(
  train_ratio = 0.7,
  val_ratio   = 0.15,
  batch_size  = 512,
  num_workers = 2L,
  seed        = 42
)

cat("✓ MNIST train_dl, val_dl, test_dl created\n")

# 3. 定义 schedule
sch_mnist <- beta_cosine(T = 1000)

# 4. 训练 UNet
cat("\nTraining UNet on MNIST training set...\n")
fit_unet <- train_diffusion_image(
  train_dl = mnist_data$train_dl,
  epochs   = 3,  # 演示用，实际可增加
  T        = 1000,
  lr       = 1e-4,
  schedule = sch_mnist,
  verbose  = TRUE,
  use_unet = TRUE,
  seed     = 42
)

# 5. 从验证集采样
cat("\nSampling from validation set...\n")
val_samples_unet <- sample_ddpm(
  fit       = fit_unet,
  n         = 16,
  steps     = 500,  # 加速采样
  shape_img = c(28, 28),
  seed      = 123
)

cat("✓ Generated", nrow(val_samples_unet), "MNIST samples\n")

# 6. 可视化生成的图像
cat("\nVisualizing generated images...\n")
par(mfrow = c(4, 4), mar = c(0.1, 0.1, 0.1, 0.1))
for (i in 1:min(16, nrow(val_samples_unet))) {
  img <- val_samples_unet[i, , ]
  image(1:28, 1:28, t(apply(img, 2, rev)),
        col = gray.colors(256), axes = FALSE)
}
par(mfrow = c(1, 1))

cat("\n✓ UNet Pipeline Complete!\n")

# ===== 度量函数快速参考 =====
cat("\n" %*% rep("=", 80))
cat("\nAvailable Metrics:")
cat("\n" %*% rep("=", 80))
cat("\n
For Distribution (MLP):
  - mmd()               : Maximum Mean Discrepancy (lower better)
  - frechet_distance()  : Frechet Distance (lower better)
  - wasserstein_1d()    : 1D Wasserstein Distance (lower better)
  - coverage_density()  : Coverage & Density metrics
  - evaluate_samples()  : Comprehensive evaluation (distribution mode)

For Images (CNN/UNet):
  - psnr()              : Peak Signal-to-Noise Ratio (higher better)
  - ssim()              : Structural Similarity (higher better)
  - evaluate_samples()  : Comprehensive evaluation (image mode)

Example usage:
  # Evaluate distribution match
  m <- evaluate_samples(X_real, X_fake, type='distribution', verbose=TRUE)
  
  # Evaluate image quality
  m <- evaluate_samples(X_real_imgs, X_fake_imgs, type='image', verbose=TRUE)
  
  # Individual metrics
  mmd_val <- mmd(X_real, X_fake, sigma=1.0)
  fd_val  <- frechet_distance(X_real, X_fake)
")

cat("\n✓ All examples completed successfully!\n\n")
