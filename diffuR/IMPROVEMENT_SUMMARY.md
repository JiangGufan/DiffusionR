<!-- 改进方案总结 -->

# DiffuR 项目改进方案 - 完整总结

## 📌 核心改进内容

您的diffusion项目已经完成了重要的改进，包括数据划分、定量评估指标和完整管道示例。

---

## ✅ 已实现的功能

### 1️⃣ 数据划分函数 (`R/datasets.R`)

#### `train_test_split(X, train_ratio=0.7, val_ratio=0.15, seed=42)`
```r
# 通用数据划分函数
splits <- train_test_split(X_std, train_ratio=0.7, val_ratio=0.15, seed=42)

# 返回：
# list(
#   train = matrix [n_train, d],    # 70% 用于训练
#   val   = matrix [n_val, d],      # 15% 用于验证模型选择
#   test  = matrix [n_test, d]      # 15% 用于最终评估
# )
```

**优点：**
- ✅ 可以检测过拟合（对比 train vs val vs test 的指标）
- ✅ 支持自定义比例和随机种子
- ✅ 返回矩阵格式，直接用于训练

---

#### `mnist_split(train_ratio=0.7, val_ratio=0.15, batch_size=128, num_workers=0, seed=42)`
```r
# MNIST 专用划分函数，直接返回 3 个 dataloader
mnist_data <- mnist_split(train_ratio=0.7, batch_size=512, num_workers=2)

# 返回：
# list(
#   train_dl = dataloader,   # 用于训练
#   val_dl = dataloader,     # 用于验证
#   test_dl = dataloader     # 用于测试
# )
```

**优点：**
- ✅ 无需手工分割，自动处理 MNIST 加载和划分
- ✅ 返回 dataloader，直接用于 `train_diffusion_image()`
- ✅ 支持多线程数据加载

---

### 2️⃣ 定量评估指标 (`R/metrics.R`)

#### 分布匹配度量

**`mmd(X1, X2, kernel_type="rbf", sigma=1.0)`** - 最大平均差异
```r
mmd_val <- mmd(X_real, X_fake, sigma=1.0)
# 返回：单个数值
# 含义：两个分布的距离，基于高斯核 RBF
# 范围：[0, ∞)，越小越好（0 表示分布完全相同）
# 时间：O(n²)，适合小数据
# 推荐：✅ 这是最常用的分布距离度量
```

**`frechet_distance(X_real, X_fake)`** - Fréchet 距离
```r
fd_val <- frechet_distance(X_real, X_fake)
# 返回：单个数值
# 含义：同时考虑均值偏移和协方差差异
# 公式：||μ₁ - μ₂||² + trace(Σ₁ + Σ₂ - 2√(Σ₁^(1/2)Σ₂Σ₁^(1/2)))
# 范围：[0, ∞)，越小越好
# 优势：更全面地反映分布差异（不仅是中心）
```

**`wasserstein_1d(X_real, X_fake)`** - Wasserstein 距离
```r
w_val <- wasserstein_1d(X_real[,1], X_fake[,1])
# 返回：单个数值
# 含义：最优运输距离，排序后的经验分布
# 范围：[0, ∞)，越小越好
# 时间：O(n log n)，很快
# 理论：有坚实的最优传输理论基础
```

#### 样本质量度量

**`coverage_density(X_real, X_fake, k=5)`** - 覆盖率和密度
```r
cov_dens <- coverage_density(X_real, X_fake, k=5)
# 返回：
# list(
#   coverage = 0.87,         # 87% 的真实样本被生成样本覆盖
#   density = 0.053,         # 生成样本到最近真实样本的平均距离
#   coverage_percent = 87    # 百分比形式
# )

# 含义：
# - coverage (0-1)：多少比例的真实数据点附近有生成样本
#                   → 衡量生成分布的多样性和覆盖能力
# - density：生成样本到最近邻真实样本的平均距离
#            → 衡量生成样本与真实数据的接近程度（质量）
```

#### 综合评估函数

**`evaluate_samples(X_real, X_fake, type="distribution", verbose=TRUE)`**
```r
# 一个函数，多个指标
metrics <- evaluate_samples(
  X_real = splits$val,
  X_fake = fake_val_std,
  type = "distribution",   # 或 "image"
  verbose = TRUE
)

# 自动输出：
# ===== Distribution Evaluation Metrics =====
# MMD:                0.127345 (lower better)
# Frechet Distance:   0.245821 (lower better)
# 1D Wasserstein:     0.089534 (lower better)
# Coverage:           88.34% (higher better)
# Density (NN dist):  0.062134 (lower better)

# 返回列表：
# list(
#   mmd = 0.127345,
#   frechet = 0.245821,
#   wasserstein_1d = 0.089534,
#   coverage = 0.8834,
#   density = 0.062134
# )
```

---

### 3️⃣ 完整工作流示例

#### 文件一览

| 文件 | 说明 | 用途 |
|------|------|------|
| `start_improved.R` | 完整的端到端示例 | 学习完整流程 |
| `example_improved_pipeline.R` | 快速集成示例 | 快速参考 |
| `IMPROVEMENT_GUIDE.md` | 详细文档 | 深入理解 |
| `QUICK_REFERENCE.md` | 快速参考 | 速查 |

#### MLP 示例流程
```r
# 1. 生成数据
X <- swiss_roll(n=2000)

# 2. 标准化
Xmean <- colMeans(X)
Xsd <- apply(X, 2, sd)
X_std <- scale(X, center=Xmean, scale=Xsd)

# 3. 划分 ✅ 新增
splits <- train_test_split(X_std, train_ratio=0.7, val_ratio=0.15, seed=42)
# train: 1400 samples
# val:   300 samples
# test:  300 samples

# 4. 训练（仅在 train 集）✅ 改进
fit <- train_diffusion_dist(
  X = splits$train,        # ← 只用训练集
  epochs = 100,
  schedule = beta_cosine(T=1000),
  verbose = TRUE
)

# 5. 验证（在 val 集评估）✅ 新增
fake_val <- sample_ddpm(fit, n=nrow(splits$val), seed=123)
evaluate_samples(
  X_real = splits$val,     # ← 验证集
  X_fake = fake_val,
  type = "distribution",
  verbose = TRUE
)

# 6. 测试（在 test 集最终评估）✅ 新增
fake_test <- sample_ddpm(fit, n=nrow(splits$test), seed=456)
evaluate_samples(
  X_real = splits$test,    # ← 测试集
  X_fake = fake_test,
  type = "distribution",
  verbose = TRUE
)

# 现在可以比较：
# - Train loss（从 trainers 输出）
# - Val metrics（第 5 步）
# - Test metrics（第 6 步）
# → 判断是否过拟合、欠拟合或泛化好
```

#### UNet 示例流程
```r
# 1. 加载并划分 MNIST ✅ 新增
mnist_data <- mnist_split(
  train_ratio = 0.7,
  val_ratio = 0.15,
  batch_size = 512,
  num_workers = 2L
)

# 2. 训练（仅在 train_dl）✅ 改进
fit_unet <- train_diffusion_image(
  train_dl = mnist_data$train_dl,  # ← 只用训练 dataloader
  epochs = 5,
  use_unet = TRUE,
  verbose = TRUE
)

# 3. 从验证集采样 ✅ 新增
val_samples <- sample_ddpm(
  fit = fit_unet,
  n = 16,
  shape_img = c(28, 28),
  seed = 123
)

# 4. 评估（图像模式）✅ 新增
# 从 val_dl 提取部分样本后
evaluate_samples(X_real_val, val_samples, type="image", verbose=TRUE)
```

---

## 📊 指标对比与解释

### 什么时候用什么指标？

#### 快速检查（训练过程中）
```r
# 用 MMD，计算快，结果稳定
mmd_val <- mmd(X_val, fake_val)
cat("Epoch", e, "| MMD:", mmd_val, "\n")
```

#### 最终评估（完整评估）
```r
# 用 evaluate_samples，一次搞定
evaluate_samples(X_test, fake_test, type="distribution", verbose=TRUE)
# 输出所有 5 个指标，自动打印
```

#### 诊断问题
```r
# 各指标组合诊断

# 情况 1: 所有指标都很好
# MMD: 0.12, Frechet: 0.25, Coverage: 90%, Density: 0.06
# → 模型很好！✅

# 情况 2: 指标太完美（都接近0）
# MMD: 0.001, Frechet: 0.002, Coverage: 100%, Density: 0.0001
# → 过拟合，只记住了训练数据 ❌

# 情况 3: Coverage 低，Density 高
# MMD: 0.30, Coverage: 40%, Density: 0.25
# → 模型没学好，样本离真实数据很远 ❌
# 对策：增加 epoch、改进架构、调学习率

# 情况 4: Coverage 好，Density 不好
# Coverage: 85%, Density: 0.15
# → 覆盖了很多真实样本，但质量不够高
# 对策：增加模型容量，用更好的架构
```

### 指标数值参考表

```
优秀模型：
┌──────────────────┬──────────┬────────┐
│ 指标             │ 范围     │ 含义   │
├──────────────────┼──────────┼────────┤
│ MMD              │ < 0.15   │ 很好匹配│
│ Frechet Distance │ < 0.3    │ 很好匹配│
│ Coverage         │ > 85%    │ 覆盖充分│
│ Density          │ < 0.08   │ 质量高 │
└──────────────────┴──────────┴────────┘

中等模型：
┌──────────────────┬──────────┬────────┐
│ 指标             │ 范围     │ 含义   │
├──────────────────┼──────────┼────────┤
│ MMD              │ 0.2-0.4  │ 还可以 │
│ Frechet Distance │ 0.4-0.7  │ 还可以 │
│ Coverage         │ 70-85%   │ 一般   │
│ Density          │ 0.08-0.15│ 一般   │
└──────────────────┴──────────┴────────┘

差的模型：
┌──────────────────┬──────────┬────────┐
│ 指标             │ 范围     │ 含义   │
├──────────────────┼──────────┼────────┤
│ MMD              │ > 0.5    │ 差异大 │
│ Frechet Distance │ > 1.0    │ 差异大 │
│ Coverage         │ < 50%    │ 覆盖不足│
│ Density          │ > 0.2    │ 质量差 │
└──────────────────┴──────────┴────────┘
```

---

## 🔑 关键改进点

### 原问题
❌ 没有 train/val/test 划分，无法判断过拟合
❌ 采样出来的数据和训练数据相同，无法区分"学到分布"vs"记住数据"
❌ 没有定量指标，只能定性比较

### 现在解决了
✅ 数据正确划分（70/15/15），可检测过拟合
✅ 定量评估指标（5种），客观衡量模型质量
✅ 完整的 train→validate→test 管道
✅ 可区分：学好了、学坏了、还是过拟合了

### 数学公式对照

#### MMD（最大平均差异）
$$\text{MMD}^2(P, Q) = \mathbb{E}_x[k(x,x')] + \mathbb{E}_y[k(y,y')] - 2\mathbb{E}_{x,y}[k(x,y)]$$

#### Frechet Distance（Fréchet 距离）
$$d_F(P, Q) = \|\mu_P - \mu_Q\|^2 + \text{trace}(\Sigma_P + \Sigma_Q - 2(\Sigma_P^{1/2}\Sigma_Q\Sigma_P^{1/2})^{1/2})$$

#### Wasserstein Distance（Wasserstein 距离，排序近似）
$$W(P, Q) \approx \frac{1}{n}\sum_{i=1}^n |x_i^{\text{sorted}} - y_i^{\text{sorted}}|$$

#### Coverage（覆盖率）
$$\text{Coverage} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}[\min_j \|x_i - y_j\| \leq \tau]$$
其中 $\tau$ 是基于 k-NN 距离的阈值

#### Density（密度）
$$\text{Density} = \frac{1}{m}\sum_{j=1}^m \min_i \|y_j - x_i\|$$

---

## 🚀 使用步骤

### 第一次使用

1. **检查文件是否已更新**
```bash
ls -la R/datasets.R
ls -la R/metrics.R
ls -la start_improved.R
```

2. **运行完整示例**
```r
source("start_improved.R")  # 会输出所有示例和结果
```

3. **查看自己的数据**
```r
# 修改示例中的数据生成部分
X <- your_custom_data()  # 替换成自己的数据
# 其余代码不变，直接运行
```

### 日常使用

```r
# 标准化的工作流

# 1. 数据划分
splits <- train_test_split(X_std)

# 2. 训练（仅用 train）
fit <- train_diffusion_dist(splits$train, epochs=100)

# 3. 验证
fake_val <- sample_ddpm(fit, n=nrow(splits$val))
m_val <- evaluate_samples(splits$val, fake_val, type="distribution")

# 4. 测试
fake_test <- sample_ddpm(fit, n=nrow(splits$test))
m_test <- evaluate_samples(splits$test, fake_test, type="distribution")

# 5. 比较 train loss（从训练输出）和 test metrics（步骤4）
#    → 判断过拟合情况
```

---

## 📚 文档导航

| 文档 | 内容 | 何时阅读 |
|------|------|--------|
| `IMPROVEMENT_GUIDE.md` | 详细、完整的说明文档 | 第一次理解项目时 |
| `QUICK_REFERENCE.md` | 速查表、常见问题 | 日常使用、遇到问题时 |
| `start_improved.R` | 完整可运行示例 | 想看具体代码时 |
| `example_improved_pipeline.R` | 快速集成示例 | 想快速上手时 |

---

## ✨ 总结

| 改进项 | 状态 | 说明 |
|--------|------|------|
| 数据划分 | ✅ | `train_test_split()` + `mnist_split()` |
| MLP 评估 | ✅ | 支持标准化分布数据 |
| CNN/UNet 评估 | ✅ | 支持 MNIST 图像数据 |
| 分布度量 | ✅ | MMD、Frechet、Wasserstein |
| 样本质量 | ✅ | Coverage、Density |
| 综合评估 | ✅ | `evaluate_samples()` 一键搞定 |
| 文档 | ✅ | 详细指南、快速参考、示例代码 |

---

## 🎓 下一步建议

1. **立即做**: 运行 `source("start_improved.R")` 看完整示例
2. **快速学**: 阅读 `QUICK_REFERENCE.md` 了解核心函数
3. **深入学**: 阅读 `IMPROVEMENT_GUIDE.md` 理解数学和细节
4. **自己试**: 用自己的数据修改示例代码，从 MLP 开始
5. **进阶用**: 尝试 UNet on MNIST，调整参数优化结果

---

**祝你使用愉快！有任何问题都可以参考文档或查看示例代码。**
