#!/usr/bin/env Rscript
# Image Demo Script for diffuR package
# Trains a diffusion model on MNIST images
# Usage: Rscript inst/benchmarks/image_demo.R

# ============================================================================
# PART 1: Load dependencies
# ============================================================================
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("Image Diffusion Demo - MNIST Training\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

library(diffuR)

# Check for torchvision
if (!requireNamespace("torchvision", quietly = TRUE)) {
  cat("WARNING: torchvision not found. Installing...\n")
  tryCatch({
    remotes::install_github("mlverse/torchvision")
  }, error = function(e) {
    cat("Could not install torchvision automatically. Please run:\n")
    cat("  remotes::install_github('mlverse/torchvision')\n")
  })
}

cat("✓ Dependencies loaded\n\n")

# ============================================================================
# PART 2: Load MNIST data
# ============================================================================
cat("Loading MNIST training data...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

tryCatch({
  dl <- mnist_train_dataloader(batch_size = 128)
  cat("✓ MNIST dataloader created successfully\n")
  cat("  - Batch size: 128\n")
  cat("  - Images: 28x28 grayscale\n")
  cat("  - Range: [0, 1]\n")
}, error = function(e) {
  cat("✗ Error loading MNIST:\n")
  cat("  ", e$message, "\n")
  stop("Cannot continue without MNIST data")
})
cat("\n")

# ============================================================================
# PART 3: Create noise schedule
# ============================================================================
cat("Creating noise schedule...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

schedule <- beta_linear(T = 400, beta_min = 1e-4, beta_max = 0.02)
cat("✓ Noise schedule created\n")
cat("  - Type: Linear\n")
cat("  - Timesteps: 400\n")
cat("  - Beta range: [", min(schedule$beta), ", ", max(schedule$beta), "]\n", sep = "")
cat("\n")

# ============================================================================
# PART 4: Train model
# ============================================================================
cat("Training diffusion model on MNIST...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

set.seed(42)
torch::torch_manual_seed(42)

fit_img <- train_diffusion_image(
  train_dl = dl,
  epochs = 1,
  T = 400,
  lr = 2e-4,
  schedule = schedule,
  verbose = TRUE,
  seed = 42
)

cat("\n✓ Training completed successfully\n")
cat("  - Model type:", fit_img$type, "\n")
cat("  - Timesteps:", fit_img$T, "\n")
cat("\n")

# ============================================================================
# PART 5: Sample new images
# ============================================================================
cat("Generating new images via sampling...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

imgs <- sample_ddpm(fit_img, n = 16, steps = 400, shape_img = c(28, 28), seed = 123)

cat("✓ Sampling completed successfully\n")
cat("  - Number of samples:", nrow(imgs), "\n")
cat("  - Image shape:", dim(imgs)[2], "x", dim(imgs)[3], "\n")
cat("  - Pixel range: [", min(imgs), ", ", max(imgs), "]\n", sep = "")
cat("\n")

# ============================================================================
# PART 6: Visualize samples
# ============================================================================
cat("Saving visualization...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

output_file <- "mnist_samples.png"

tryCatch({
  png(output_file, width = 560, height = 560)
  par(mfrow = c(4, 4), mar = c(0, 0, 0, 0))
  for (i in 1:16) {
    image(imgs[i, , ], col = gray.colors(256), axes = FALSE)
  }
  dev.off()
  cat("✓ Visualization saved to:", output_file, "\n")
}, error = function(e) {
  cat("✗ Error saving visualization:\n")
  cat("  ", e$message, "\n")
})

cat("\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("Image demo completed!\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
