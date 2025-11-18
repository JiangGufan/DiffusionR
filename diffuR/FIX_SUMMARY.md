# 🔧 快速修复指南 - CNN 模型错误

## 问题

```
Error: expected input[1, 29, 28, 128] to have 2 channels, but got 29 channels
```

## 原因

`t_frac` 形状不正确导致张量拼接时通道数错误。

## 修复

### 1. `R/trainers.R` - 第 62 行

**改变这一行:**
```r
# ❌ 错误
t_frac <- torch::torch_tensor(matrix(t_idx / T, bs, 1))

# ✅ 正确
t_frac <- torch::torch_tensor(as.numeric(t_idx) / T, dtype = torch::torch_float())
```

### 2. `R/model_cnn.R` - forward 函数

**改为:**
```r
forward = function(x, t_frac){
  B <- x$size()[1]; H <- x$size()[3]; W <- x$size()[4]
  
  # 处理形状
  if (length(t_frac$shape) == 1) {
    t_frac <- t_frac$unsqueeze(2)
  }
  
  em <- self$emb(t_frac)
  em <- em$view(c(B,1,1,1))$expand(c(B,1,H,W))
  x_in <- torch::torch_cat(list(x, em), dim = 2)
  
  h <- torch::nnf_relu(self$conv1(x_in))
  h <- torch::nnf_relu(self$conv2(h))
  out <- self$conv3(h)
  out
}
```

## 测试

```bash
# 重新编译
R CMD INSTALL --no-multiarch --with-keep.source .

# 运行测试
Rscript test_cnn_fix.R

# 现在可以训练了!
Rscript -e "
  library(devtools)
  load_all('.')
  
  dl <- mnist_train_dataloader(batch_size = 4)
  fit <- train_diffusion_image(
    train_dl = dl,
    epochs = 1,
    T = 100,
    schedule = beta_linear(T = 100)
  )
  cat('✓ 成功!\n')
"
```

## ✅ 已修复!

现在可以正常训练 CNN 模型了。
