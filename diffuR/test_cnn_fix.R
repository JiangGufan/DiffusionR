#!/usr/bin/env Rscript
# Test script to verify CNN model fix
# 测试 CNN 模型修复

library(devtools)

# 加载包
pkg_root <- dirname(dirname(getwd()))
load_all(pkg_root)

cat("=" %+% rep("=", 78) %+% "\n", sep = "")
cat("Testing CNN Model Fix\n")
cat("=" %+% rep("=", 78) %+% "\n\n")

# 准备小型 MNIST 数据
cat("1. 准备 MNIST 数据...\n")
tryCatch({
  train_dl <- mnist_train_dataloader(batch_size = 4)  # 很小的批处理用于测试
  cat("   ✓ 数据加载成功\n\n")
}, error = function(e) {
  cat("   ✗ 数据加载失败:", e$message, "\n")
  stop()
})

# 构建模型
cat("2. 构建 CNN 模型...\n")
tryCatch({
  model <- build_cnn_eps(ch = 32)()
  cat("   ✓ 模型构建成功\n\n")
}, error = function(e) {
  cat("   ✗ 模型构建失败:", e$message, "\n")
  stop()
})

# 创建时间表
cat("3. 创建噪声时间表...\n")
schedule <- beta_linear(T = 100)
cat("   ✓ 时间表创建成功\n\n")

# 测试一个 batch
cat("4. 测试前向传播 (单个batch)...\n")
tryCatch({
  batch <- first(train_dl)  # 获取第一个 batch
  x0 <- batch[[1]]$to(dtype = torch::torch_float()) * 2 - 1
  bs <- x0$size()[1]
  
  cat(paste0("   - 批处理大小: ", bs, "\n"))
  cat(paste0("   - 输入张量形状: ", paste(as.numeric(x0$shape), collapse=" x "), "\n"))
  
  # 随机时间
  t_idx <- sample.int(100, bs, replace = TRUE)
  t_frac <- torch::torch_tensor(as.numeric(t_idx) / 100, dtype = torch::torch_float())
  
  cat(paste0("   - 时间张量形状: ", paste(as.numeric(t_frac$shape), collapse=" x "), "\n"))
  
  # 前向传播
  eps_pred <- model(x0, t_frac)
  
  cat(paste0("   - 输出张量形状: ", paste(as.numeric(eps_pred$shape), collapse=" x "), "\n"))
  cat("   ✓ 前向传播成功!\n\n")
}, error = function(e) {
  cat("   ✗ 前向传播失败:", e$message, "\n")
  cat("   详细错误信息:\n")
  print(e)
  stop()
})

# 测试训练步骤
cat("5. 测试训练步骤 (1 个 batch)...\n")
tryCatch({
  opt <- torch::optim_adam(model$parameters, lr = 2e-4)
  model$train()
  
  coro::loop(for (b in train_dl) {
    x0 <- b[[1]]$to(dtype = torch::torch_float()) * 2 - 1
    bs <- x0$size()[1]
    t_idx <- sample.int(100, bs, replace = TRUE)
    t_frac <- torch::torch_tensor(as.numeric(t_idx) / 100, dtype = torch::torch_float())
    eps_true <- torch::torch_randn_like(x0)
    
    # 获取预测
    eps_pred <- model(x0, t_frac)
    
    # 计算损失
    loss <- torch::nnf_mse_loss(eps_pred, eps_true, reduction = "mean")
    
    # 反向传播
    opt$zero_grad()
    loss$backward()
    opt$step()
    
    cat(paste0("   - 损失: ", sprintf("%.6f", as.numeric(loss$item())), "\n"))
    break  # 只测试一个 batch
  })
  cat("   ✓ 训练步骤成功!\n\n")
}, error = function(e) {
  cat("   ✗ 训练步骤失败:", e$message, "\n")
  cat("   详细错误信息:\n")
  print(e)
  stop()
})

# 快速训练测试
cat("6. 快速训练测试 (1 epoch, 小数据)...\n")
tryCatch({
  fit <- train_diffusion_image(
    train_dl = train_dl,
    epochs = 1,
    T = 100,
    lr = 2e-4,
    schedule = schedule,
    verbose = TRUE,
    seed = 42
  )
  cat("   ✓ 训练成功!\n\n")
}, error = function(e) {
  cat("   ✗ 训练失败:", e$message, "\n")
  cat("   详细错误信息:\n")
  print(e)
  stop()
})

cat("=" %+% rep("=", 78) %+% "\n", sep = "")
cat("所有测试通过! ✓\n")
cat("=" %+% rep("=", 78) %+% "\n")
