# 快速参考指南 - diffuR Package 测试

## 🎯 核心测试脚本

### 选项 1: 完整单元测试 (推荐首先运行)
```bash
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR
Rscript inst/test_demo.R
```
**用途**: 验证所有核心功能是否正常工作
- ✓ 检查导出的函数
- ✓ 创建数据集
- ✓ 构建模型 (MLP & CNN)
- ✓ 创建噪声时间表
- ✓ 训练模型 (2个epoch的快速测试)
- ✓ 采样生成
- ✓ 可视化

### 选项 2: Swiss Roll 分布演示
```bash
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR
Rscript inst/benchmarks/distribution_demo.R
```
**用途**: 在 2D 分布数据上训练并采样
- 生成 4000 个 Swiss Roll 样本
- 训练 15 个 epoch
- 采样 4000 个新样本
- 输出: `dist_demo.png` (对比图)

### 选项 3: MNIST 图像演示
```bash
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR
Rscript inst/benchmarks/image_demo.R
```
**用途**: 在 MNIST 图像上训练并采样
- 从 torchvision 下载 MNIST
- 训练 1 个 epoch (快速演示)
- 采样 16 个图像
- 输出: `mnist_samples.png` (4×4 网格)

---

## 📊 调用文件链接关系

```
test_demo.R
├── R/api.R (空,占位符)
├── R/datasets.R
│   ├── swiss_roll()
│   ├── plot_2d_samples()
│   └── mnist_train_dataloader()
├── R/schedules.R
│   ├── beta_linear()
│   └── beta_cosine()
├── R/model_mlp.R
│   └── build_mlp_eps()
├── R/model_cnn.R
│   └── build_cnn_eps()
├── R/trainers.R
│   ├── train_diffusion_dist()
│   └── train_diffusion_image()
└── R/sampling.R
    ├── sample_ddpm()
    └── sample_vp_sde()

distribution_demo.R → [datasets, schedules, trainers, sampling]
image_demo.R → [datasets, schedules, trainers, sampling]
```

---

## 🔧 完整工作流程

### 1️⃣ 分布模型流程
```r
# 创建数据
X <- swiss_roll(n = 4000, noise = 0.15)

# 创建时间表
schedule <- beta_linear(T = 500)

# 训练
fit <- train_diffusion_dist(X, epochs = 15, T = 500, schedule = schedule)

# 采样
samples <- sample_ddpm(fit, n = 4000, steps = 500)

# 可视化
p <- plot_2d_samples(X, samples)
```

### 2️⃣ 图像模型流程
```r
# 加载数据
dl <- mnist_train_dataloader(batch_size = 128)

# 创建时间表
schedule <- beta_linear(T = 400)

# 训练
fit <- train_diffusion_image(dl, epochs = 1, T = 400, schedule = schedule)

# 采样
imgs <- sample_ddpm(fit, n = 16, steps = 400, shape_img = c(28, 28))

# 可视化
png("result.png", width = 560, height = 560)
par(mfrow = c(4, 4))
for (i in 1:16) image(imgs[i,,], col = gray.colors(256))
dev.off()
```

---

## 📦 安装/编译

### 编译 C++ 代码
```bash
# 在 package 根目录
cd /Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR
R CMD INSTALL --no-multiarch --with-keep.source .
```

### 从 R 编译
```r
devtools::install(pkg = "/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR")
```

---

## ✅ 验收标准

| 项目 | 预期 |
|------|------|
| 函数可访问 | ✓ 所有导出函数可调用 |
| 数据创建 | ✓ swiss_roll() 成功 |
| 模型构建 | ✓ MLP/CNN 初始化无错 |
| 训练 | ✓ 损失函数递减 |
| 采样 | ✓ 生成有效样本 |
| 可视化 | ✓ 生成 PNG 文件 |

---

## 🐛 故障排除

### "找不到模块" 错误
```r
torch::torch_version()  # 检查 torch
# 如果不行，重新安装:
remotes::install_github("mlverse/torch")
```

### MNIST 下载失败
- 脚本会自动重试
- 检查网络连接
- 手动下载后指定路径

### 内存不足
```r
# 减小参数
swiss_roll(n = 1000)      # 减少样本
batch_size = 128          # 减小批处理
T = 100                   # 减少时间步
```

---

## 📁 输出文件

| 文件 | 位置 | 描述 |
|------|------|------|
| `dist_demo.png` | 当前工作目录 | Swiss Roll 对比 |
| `mnist_samples.png` | 当前工作目录 | MNIST 生成的图像 |
| `test_plot.png` | diffuR 根目录 | 单元测试可视化 |

---

## 🚀 推荐执行顺序

1. **首次测试**: `Rscript inst/test_demo.R` (快速验证)
2. **功能演示**: `Rscript inst/benchmarks/distribution_demo.R` (展示2D能力)
3. **高级演示**: `Rscript inst/benchmarks/image_demo.R` (展示图像能力)

---

**版本**: 1.0  
**最后更新**: 2025年11月13日
