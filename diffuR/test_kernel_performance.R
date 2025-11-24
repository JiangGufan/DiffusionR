# ============================================
# Kernel集成性能测试脚本
# 验证C++ kernel加速效果
# ============================================

library(Rcpp)
library(torch)
library(coro)

# ✅ 确保kernel已编译
Rcpp::sourceCpp("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/src/kernels.cpp")

# 加载模块
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/datasets.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/schedules.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/model_mlp.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/trainers.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/sampling.R")

cat("========================================\n")
cat("C++ Kernel Integration Performance Test\n")
cat("========================================\n\n")

# ============= 测试 1：q_sample_xt_given_x0 =============
cat("Test 1: q_sample_xt_given_x0 Performance\n")
cat("-----------------------------------------\n")

X <- matrix(rnorm(2000 * 2), 2000, 2)  # 2000 samples, 2D
Xmean <- colMeans(X)
Xsd <- apply(X, 2, sd)
X_std <- scale(X, center = Xmean, scale = Xsd)

schedule <- beta_cosine(T = 1000)
t_test <- 500

# 方法1：原始 torch 手写
cat("Method 1: Torch manual (手写torch张量)...\n")
time_torch <- system.time({
  for (rep in 1:100) {
    sa <- torch::torch_tensor(matrix(schedule$sqrt_alpha_bar[t_test], nrow(X_std), 1))
    som <- torch::torch_tensor(matrix(schedule$sqrt_one_minus_alpha_bar[t_test], nrow(X_std), 1))
    eps_true <- torch::torch_randn(nrow(X_std), ncol(X_std))
    x_t <- sa * X_std + som * eps_true
    rm(sa, som, eps_true, x_t)
  }
})

# 方法2：使用 C++ kernel
cat("Method 2: C++ Kernel (q_sample_xt_given_x0)...\n")
time_kernel <- system.time({
  for (rep in 1:100) {
    x_t_mat <- q_sample_xt_given_x0(
      X0 = as.matrix(X_std),
      sa = schedule$sqrt_alpha_bar[t_test],
      om = schedule$sqrt_one_minus_alpha_bar[t_test]
    )
    x_t <- torch::torch_tensor(x_t_mat)
    rm(x_t_mat, x_t)
  }
})

cat("\nResults:\n")
cat(sprintf("  Torch:  %.3f sec (elapsed)\n", time_torch["elapsed"]))
cat(sprintf("  Kernel: %.3f sec (elapsed)\n", time_kernel["elapsed"]))
cat(sprintf("  Speedup: %.2f x\n\n", time_torch["elapsed"] / time_kernel["elapsed"]))

# ============= 测试 2：train_diffusion_dist 端到端 =============
cat("Test 2: Full Training with Kernel Integration\n")
cat("---------------------------------------------\n")

X_small <- swiss_roll(n = 500, noise = 0.1, seed = 42)
Xmean <- colMeans(X_small)
Xsd <- apply(X_small, 2, sd)
X_std_small <- scale(X_small, center = Xmean, scale = Xsd)

cat("Training on Swiss Roll (500 samples, 10 epochs)...\n")
time_train <- system.time({
  fit <- train_diffusion_dist(
    X = X_std_small,
    epochs = 10,
    T = 100,      # 短T加快测试
    lr = 1e-3,
    batch_size = 128,
    schedule = beta_linear(T = 100),
    verbose = FALSE,
    seed = 42
  )
})

cat(sprintf("Training time: %.3f sec\n", time_train["elapsed"]))
cat(sprintf("Per epoch: %.3f sec\n\n", time_train["elapsed"] / 10))

# ============= 测试 3：采样性能对比 =============
cat("Test 3: Sampling with Kernel (sample_ddpm)\n")
cat("-----------------------------------------\n")

cat("Sampling 100 samples, 50 steps...\n")
time_sample <- system.time({
  samples <- sample_ddpm(
    fit = fit,
    n = 100,
    steps = 50,  # 短采样加快测试
    seed = 42
  )
})

cat(sprintf("Sampling time: %.3f sec\n", time_sample["elapsed"]))
cat(sprintf("Per sample: %.3f msec\n\n", time_sample["elapsed"] / 100 * 1000))

# ============= 测试 4：ddpm_posterior 性能 =============
cat("Test 4: Posterior Computation Performance\n")
cat("-----------------------------------------\n")

n_test <- 500
d_test <- 2
t_idx <- 500

# 生成测试数据
x_mat <- matrix(rnorm(n_test * d_test), n_test, d_test)
eps_mat <- matrix(rnorm(n_test * d_test), n_test, d_test)
schedule_test <- beta_cosine(T = 1000)

cat(sprintf("Computing posterior for %d samples, dimension %d, 100 times...\n", 
            n_test, d_test))

# 方法1：手写（模拟原来的方式）
cat("Method 1: Manual computation...\n")
time_manual <- system.time({
  for (rep in 1:100) {
    alpha_t <- schedule_test$alpha[t_idx]
    alpha_bar_t <- schedule_test$alpha_bar[t_idx]
    beta_t <- schedule_test$beta[t_idx]
    alpha_bar_prev <- schedule_test$alpha_bar[max(1, t_idx - 1)]
    
    mu_manual <- (1 / sqrt(alpha_t)) * (
      x_mat - (beta_t / sqrt(1 - alpha_bar_t)) * eps_mat
    )
    var_manual <- (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
  }
})

# 方法2：C++ kernel
cat("Method 2: C++ Kernel (ddpm_posterior_*)...\n")
time_posterior_kernel <- system.time({
  for (rep in 1:100) {
    mu_kernel <- ddpm_posterior_mean(
      x_t = x_mat,
      eps_pred = eps_mat,
      sqrt_alpha_bar_t = rep(sqrt(schedule_test$alpha_bar[t_idx]), n_test),
      sqrt_alpha_bar_tm1 = rep(sqrt(schedule_test$alpha_bar[max(1, t_idx - 1)]), n_test),
      beta_t = rep(schedule_test$beta[t_idx], n_test),
      one_minus_alpha_bar_t = rep(1 - schedule_test$alpha_bar[t_idx], n_test),
      one_minus_alpha_bar_tm1 = rep(1 - schedule_test$alpha_bar[max(1, t_idx - 1)], n_test)
    )
    var_kernel <- ddpm_posterior_var(
      beta_t = rep(schedule_test$beta[t_idx], n_test),
      one_minus_alpha_bar_t = rep(1 - schedule_test$alpha_bar[t_idx], n_test),
      one_minus_alpha_bar_tm1 = rep(1 - schedule_test$alpha_bar[max(1, t_idx - 1)], n_test)
    )
  }
})

cat("\nResults:\n")
cat(sprintf("  Manual:  %.3f sec\n", time_manual["elapsed"]))
cat(sprintf("  Kernel:  %.3f sec\n", time_posterior_kernel["elapsed"]))
cat(sprintf("  Speedup: %.2f x\n\n", time_manual["elapsed"] / time_posterior_kernel["elapsed"]))

# ============= 汇总 =============
cat("========================================\n")
cat("SUMMARY: Performance Improvements\n")
cat("========================================\n")

q_sample_speedup <- time_torch["elapsed"] / time_kernel["elapsed"]
posterior_speedup <- time_manual["elapsed"] / time_posterior_kernel["elapsed"]

cat(sprintf("1. q_sample_xt_given_x0:     %.2f x faster\n", q_sample_speedup))
cat(sprintf("2. ddpm_posterior_* (100x):   %.2f x faster\n", posterior_speedup))
cat(sprintf("\nEstimated overall training speedup (2D data): %.2f x\n", 
            (q_sample_speedup + posterior_speedup) / 2))

cat("\nConclusions:\n")
if (q_sample_speedup > 1.2 && posterior_speedup > 1.2) {
  cat("✅ C++ Kernel integration is SUCCESSFUL and brings significant speedup\n")
  cat("   → Use kernels for low-dimensional diffusion operations\n")
  cat("   → Use torch for neural network gradients\n")
} else {
  cat("⚠️ Speedup is marginal. Consider:\n")
  cat("   → Data dimension is very small (too little computation)\n")
  cat("   → Overhead of R-C++ conversion not worth it\n")
  cat("   → For production, use GPU-accelerated tensors\n")
}

cat("\n========================================\n")
