#!/usr/bin/env Rscript
# Distribution Demo Script for diffuR package
# Trains a diffusion model on Swiss Roll dataset
# Usage: Rscript inst/benchmarks/distribution_demo.R

# ============================================================================
# PART 1: Load dependencies
# ============================================================================
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("Distribution Diffusion Demo - Swiss Roll\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

library(diffuR)

# Check for ggplot2
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  cat("WARNING: ggplot2 not found. Installing...\n")
  install.packages("ggplot2")
}

cat("✓ Dependencies loaded\n\n")

# ============================================================================
# PART 2: Load data
# ============================================================================
cat("Creating Swiss Roll dataset...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

set.seed(1)
X <- swiss_roll(n = 4000, noise = 0.15, seed = 1)

cat("✓ Swiss Roll created successfully\n")
cat("  - Number of samples:", nrow(X), "\n")
cat("  - Dimension:", ncol(X), "\n")
cat("  - X1 range: [", min(X[,1]), ", ", max(X[,1]), "]\n", sep = "")
cat("  - X2 range: [", min(X[,2]), ", ", max(X[,2]), "]\n", sep = "")
cat("\n")

# ============================================================================
# PART 3: Create noise schedule
# ============================================================================
cat("Creating noise schedule...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

schedule <- beta_linear(T = 500, beta_min = 1e-4, beta_max = 0.02)

cat("✓ Noise schedule created\n")
cat("  - Type: Linear\n")
cat("  - Timesteps: 500\n")
cat("  - Beta range: [", min(schedule$beta), ", ", max(schedule$beta), "]\n", sep = "")
cat("\n")

# ============================================================================
# PART 4: Train model
# ============================================================================
cat("Training diffusion model on Swiss Roll...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

set.seed(1)
torch::torch_manual_seed(1)

fit <- train_diffusion_dist(
  X = X,
  epochs = 15,
  T = 500,
  lr = 1e-3,
  batch_size = 1024,
  schedule = schedule,
  verbose = TRUE,
  seed = 1
)

cat("\n✓ Training completed successfully\n")
cat("  - Model type:", fit$type, "\n")
cat("  - Data dimension:", fit$dim, "\n")
cat("  - Timesteps:", fit$T, "\n")
cat("\n")

# ============================================================================
# PART 5: Generate samples
# ============================================================================
cat("Generating samples from trained model...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

samp <- sample_ddpm(fit, n = 4000, steps = 500, seed = 123)
colnames(samp) <- colnames(X)

cat("✓ Sampling completed successfully\n")
cat("  - Number of samples:", nrow(samp), "\n")
cat("  - X1 range: [", min(samp[,1]), ", ", max(samp[,1]), "]\n", sep = "")
cat("  - X2 range: [", min(samp[,2]), ", ", max(samp[,2]), "]\n", sep = "")
cat("\n")

# ============================================================================
# PART 6: Create visualization
# ============================================================================
cat("Creating comparison plot...\n")
cat(paste0(rep("-", 80), collapse = ""), "\n")

tryCatch({
  p <- plot_2d_samples(X, samp)
  
  output_file <- "dist_demo.png"
  ggplot2::ggsave(
    filename = output_file,
    plot = p,
    width = 10,
    height = 5,
    dpi = 150
  )
  
  cat("✓ Visualization saved to:", output_file, "\n")
  cat("  - Left: Original Swiss Roll data (", nrow(X), " samples)\n", sep = "")
  cat("  - Right: Generated samples from diffusion model (", nrow(samp), " samples)\n", sep = "")
}, error = function(e) {
  cat("✗ Error creating visualization:\n")
  cat("  ", e$message, "\n")
})

cat("\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("Distribution demo completed!\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
