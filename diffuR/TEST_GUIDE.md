# 📊 diffuR Package Testing Guide (R 脚本调用指南)

## 概述

这份指南说明如何在封装 R package 之前使用 Rscript 脚本来测试 diffusion model 的构造是否正确。

---

## 📁 目录结构

```
diffuR/
├── inst/
│   ├── test_demo.R                    ✓ 完整单元测试脚本
│   └── benchmarks/
│       ├── distribution_demo.R        ✓ Swiss Roll 分布演示
│       └── image_demo.R               ✓ MNIST 图像演示
├── R/
│   ├── api.R                          API 包装器
│   ├── datasets.R                     数据集函数
│   ├── metrics.R                      评估指标
│   ├── model_cnn.R                    CNN 模型
│   ├── model_mlp.R                    MLP 模型
│   ├── parallel.R                     并行计算
│   ├── sampling.R                     采样函数
│   ├── schedules.R                    噪声时间表
│   └── trainers.R                     训练函数
└── src/
    ├── init.c                         Rcpp 初始化
    ├── kernels.cpp                    C++ 核心内核
    ├── Makevars                       Linux/macOS 编译配置
    └── Makevars.win                   Windows 编译配置
```

---

## 🚀 快速开始

### 前置条件

确保已安装以下依赖：

```r
# 安装必要的 R 包
install.packages(c("torch", "Rcpp", "RcppArmadillo", "ggplot2"))

# 可选：MNIST 演示需要
install.packages("remotes")
remotes::install_github("mlverse/torchvision")
```

### 编译 Package

在 package 根目录运行：

```bash
# 编译 C++ 代码
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR
R CMD INSTALL --no-multiarch --with-keep.source .
```

或在 R 中：

```r
devtools::install(pkg = "/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR")
```

---

## 🧪 测试脚本使用方法

### 1️⃣ **TEST DEMO** - 完整单元测试

**目的**: 验证所有核心功能是否正常工作

**运行方式**:

```bash
# 方法 1: 从 package 根目录
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR
Rscript inst/test_demo.R

# 方法 2: 从任意位置使用完整路径
Rscript /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/inst/test_demo.R
```

**输出示例**:
```
================================================================================
Loading diffuR package in development mode...
================================================================================

TEST 1: Checking exported functions...
────────────────────────────────────────────────────────────────────────────
✓ All expected functions are available

TEST 2: Creating sample dataset (Swiss Roll)...
────────────────────────────────────────────────────────────────────────────
✓ Swiss roll created successfully
  - Shape: 500 x 2
  - X1 range: [-0.123, 0.456]
  - X2 range: [-0.789, 0.321]

TEST 3: Building neural network models...
────────────────────────────────────────────────────────────────────────────
✓ MLP epsilon-predictor created successfully
  - Input dimension: 2 (data) + 1 (time) = 3
  - Hidden dimension: 128
  - Output dimension: 2

✓ CNN epsilon-predictor created successfully
  - Channel dimension: 32
  - Input: (B, 1, 28, 28) + (B, 1, 28, 28) time embedding = (B, 2, 28, 28)
  - Output: (B, 1, 28, 28)

TEST 4: Creating noise schedules...
────────────────────────────────────────────────────────────────────────────
✓ Linear schedule created successfully (T=100)
✓ Cosine schedule created successfully (T=100)

TEST 5: Training diffusion model (quick test - 2 epochs)...
────────────────────────────────────────────────────────────────────────────
Epoch 1 | loss=0.8234
Epoch 2 | loss=0.5123
✓ Training completed successfully

TEST 6: Sampling from trained diffusion model...
────────────────────────────────────────────────────────────────────────────
✓ Sampling completed successfully
  - Number of samples: 50
  - Sample dimension: 2

TEST 7: Creating visualization...
────────────────────────────────────────────────────────────────────────────
✓ Visualization saved to: test_plot.png

================================================================================
All tests completed!
================================================================================
```

**测试覆盖范围**:
- ✓ 函数导出检查
- ✓ 数据集创建 (Swiss Roll)
- ✓ 神经网络模型构建 (MLP & CNN)
- ✓ 噪声时间表 (Linear & Cosine)
- ✓ 模型训练
- ✓ 样本生成
- ✓ 可视化

---

### 2️⃣ **DISTRIBUTION DEMO** - Swiss Roll 分布演示

**目的**: 演示在 2D 分布上训练和采样

**运行方式**:

```bash
# 方法 1: 从 benchmarks 目录
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/inst/benchmarks
Rscript distribution_demo.R

# 方法 2: 从 package 根目录
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR
Rscript inst/benchmarks/distribution_demo.R
```

**输出文件**: `dist_demo.png`
- 左侧: 原始 Swiss Roll 数据 (4000 样本)
- 右侧: 扩散模型生成的样本 (4000 样本)

**输出示例**:
```
================================================================================
Distribution Diffusion Demo - Swiss Roll
================================================================================

Creating Swiss Roll dataset...
────────────────────────────────────────────────────────────────────────────
✓ Swiss Roll created successfully
  - Number of samples: 4000
  - Dimension: 2

Training diffusion model on Swiss Roll...
────────────────────────────────────────────────────────────────────────────
Epoch 1 | loss=0.9523
Epoch 2 | loss=0.8234
...
Epoch 15 | loss=0.1234
✓ Training completed successfully

Generating samples from trained model...
────────────────────────────────────────────────────────────────────────────
✓ Sampling completed successfully

Creating comparison plot...
────────────────────────────────────────────────────────────────────────────
✓ Visualization saved to: dist_demo.png
```

**参数说明**:
- `n = 4000`: 训练样本数
- `noise = 0.15`: 数据噪声水平
- `epochs = 15`: 训练轮数
- `T = 500`: 扩散时间步数
- `batch_size = 1024`: 批处理大小

---

### 3️⃣ **IMAGE DEMO** - MNIST 图像演示

**目的**: 演示在 MNIST 图像数据集上训练和采样

**运行方式**:

```bash
# 从 benchmarks 目录
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/inst/benchmarks
Rscript image_demo.R

# 或从 package 根目录
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR
Rscript inst/benchmarks/image_demo.R
```

**输出文件**: `mnist_samples.png`
- 4×4 网格显示 16 个生成的 28×28 MNIST 数字图像

**输出示例**:
```
================================================================================
Image Diffusion Demo - MNIST Training
================================================================================

Loading MNIST training data...
────────────────────────────────────────────────────────────────────────────
✓ MNIST dataloader created successfully
  - Batch size: 128
  - Images: 28x28 grayscale

Training diffusion model on MNIST...
────────────────────────────────────────────────────────────────────────────
Epoch 1 | loss=0.4523
✓ Training completed successfully

Generating samples from trained model...
────────────────────────────────────────────────────────────────────────────
✓ Sampling completed successfully
  - Number of samples: 16

Saving visualization...
────────────────────────────────────────────────────────────────────────────
✓ Visualization saved to: mnist_samples.png
```

**参数说明**:
- `batch_size = 128`: 批处理大小
- `epochs = 1`: 训练轮数（快速演示）
- `T = 400`: 扩散时间步数
- `n = 16`: 生成的图像数量

---

## 📋 各文件的职责

| 文件 | 职责 | 导出函数 |
|-----|------|--------|
| `R/api.R` | API 包装器 | (占位符) |
| `R/datasets.R` | 数据集生成与加载 | `swiss_roll()`, `plot_2d_samples()`, `mnist_train_dataloader()` |
| `R/metrics.R` | 评估指标 | (内部使用) |
| `R/model_cnn.R` | CNN 模型 | `build_cnn_eps()` |
| `R/model_mlp.R` | MLP 模型 | `build_mlp_eps()` |
| `R/schedules.R` | 噪声时间表 | `beta_linear()`, `beta_cosine()` |
| `R/sampling.R` | 采样函数 | `sample_ddpm()`, `sample_vp_sde()`, `score_from_eps()` |
| `R/trainers.R` | 训练函数 | `train_diffusion_dist()`, `train_diffusion_image()` |
| `src/kernels.cpp` | C++ 核心 | `vp_reverse_sde_step()` |

---

## 🔧 完整调用流程

### 流程 1: 分布演示流程

```r
# 1. 创建数据
X <- swiss_roll(n = 4000, noise = 0.15)

# 2. 创建时间表
schedule <- beta_linear(T = 500)

# 3. 训练模型
fit <- train_diffusion_dist(
  X = X,
  epochs = 15,
  T = 500,
  schedule = schedule
)

# 4. 生成样本
samples <- sample_ddpm(fit, n = 4000, steps = 500)

# 5. 可视化
p <- plot_2d_samples(X, samples)
ggplot2::ggsave("result.png", p)
```

### 流程 2: 图像演示流程

```r
# 1. 加载数据
dl <- mnist_train_dataloader(batch_size = 128)

# 2. 创建时间表
schedule <- beta_linear(T = 400)

# 3. 训练模型
fit <- train_diffusion_image(
  train_dl = dl,
  epochs = 1,
  T = 400,
  schedule = schedule
)

# 4. 生成样本
imgs <- sample_ddpm(fit, n = 16, steps = 400, shape_img = c(28, 28))

# 5. 可视化
png("result.png", width = 560, height = 560)
par(mfrow = c(4, 4))
for (i in 1:16) image(imgs[i,,], col = gray.colors(256))
dev.off()
```

---

## ⚠️ 常见问题

### Q1: "找不到模块" 错误
```
Error: 找不到 C++ 编译器或 torch 包
```
**解决**:
```r
# 检查 torch 是否正确安装
torch::torch_version()

# 重新安装
remotes::install_github("mlverse/torch")
```

### Q2: "cuda/device" 错误
```
Error: No CUDA device found
```
**解决**: 脚本会自动使用 CPU，无需手动配置

### Q3: MNIST 下载失败
```
Error: Cannot download MNIST
```
**解决**:
```r
# 手动指定临时目录
torchvision::mnist_dataset(root = "~/Downloads/mnist", download = TRUE)
```

### Q4: 内存不足
**解决**: 减少参数
```r
# 减少样本数、批处理大小或时间步
X_small <- swiss_roll(n = 1000, noise = 0.15)
train_diffusion_dist(X_small, batch_size = 256, T = 100)
```

---

## ✅ 验收标准

| 项目 | 预期结果 |
|------|--------|
| 所有函数可访问 | ✓ 所有导出函数都能调用 |
| 数据集创建 | ✓ 无错误生成数据 |
| 模型构建 | ✓ MLP 和 CNN 都能成功初始化 |
| 训练过程 | ✓ 损失函数递减 |
| 采样 | ✓ 生成有效的样本 |
| 可视化 | ✓ 生成对比图像 |

---

## 📝 快速参考命令

```bash
# 编译
R CMD INSTALL --no-multiarch --with-keep.source /path/to/diffuR

# 完整测试
Rscript /path/to/diffuR/inst/test_demo.R

# 分布演示
Rscript /path/to/diffuR/inst/benchmarks/distribution_demo.R

# 图像演示
Rscript /path/to/diffuR/inst/benchmarks/image_demo.R
```

---

## 🎯 下一步

- ✅ 所有测试通过后，准备发布 package
- 📦 运行 `R CMD check`
- 📖 生成文档 `devtools::document()`
- 🔍 检查代码风格 `lintr::lint_package()`

---

**更新日期**: 2025年11月13日
**Status**: ✅ 所有演示脚本已完成并调试
