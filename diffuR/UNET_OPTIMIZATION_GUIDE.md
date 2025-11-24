# DDPM 模型优化指南：从简单CNN到UNet

## 问题分析

你的原始4层CNN模型在MNIST图像生成上效果较差的原因：

| 问题 | 影响 |
|------|------|
| **架构过浅** | 缺少深层次的特征层次化 |
| **无跳连接** | 梯度消失，信息流动差 |
| **无多尺度处理** | 无法同时捕捉局部细节和全局结构 |
| **时间编码弱** | 对不同扩散时间步的区分能力有限 |
| **无注意力机制** | 无法学习像素间的长程依赖关系 |

---

## 新增架构：UNet

### 核心设计特性

#### 1. **残差块（ResNet Block）**
```r
resnet_block(in_ch, out_ch, t_emb_dim)
```
- **双卷积** + **残差连接**：缓解梯度消失
- **GroupNorm**：比BatchNorm更稳定（不依赖batch size）
- **时间嵌入集成**：在每一层融入扩散时间信息
- **SiLU激活**：比ReLU更平滑的梯度

#### 2. **注意力块（Attention Block）**
```r
attention_block(ch)
```
- **Self-Attention**：学习像素间的长程依赖
- **特别重要**：MNIST虽小，但笔迹的连贯性依赖长程依赖
- 计算复杂度：$O((HW)^2)$，对28×28可承受

#### 3. **UNet结构**
```
输入 [B,1,28,28]
   ↓
Input Conv → [B,64,28,28]
   ↓
Encoder (下采样):
  Level 1: ResBlock + Attention → [B,64,28,28]
           ↓ (stride=2)
  Level 2: ResBlock + Attention → [B,128,14,14]
           ↓ (stride=2)
  Level 3: ResBlock + Attention → [B,256,7,7]
   ↓
中间瓶颈: ResBlock → Attention → ResBlock → [B,256,7,7]
   ↓
Decoder (上采样 + Skip Connection):
  Level 3: ↑ (stride=2) → [B,256,14,14]
           Cat skip → ResBlock + Attention
  Level 2: ↑ (stride=2) → [B,128,28,28]
           Cat skip → ResBlock + Attention
  Level 1: ResBlock + Attention → [B,64,28,28]
   ↓
Output Conv → [B,1,28,28]
```

### 参数对比

| 架构 | 参数数 | 推理时间 | 生成质量 |
|------|--------|---------|---------|
| 简单CNN | ~200K | 快 | 差 |
| **UNet (推荐)** | **~5M** | 中等 | **优秀** |
| UNet (大) | 20M+ | 慢 | 最优 |

---

## 使用方法

### 1. 使用UNet训练（默认）

```r
# 直接使用 UNet（推荐）
fit_img <- train_diffusion_image(
  train_dl = train_dl,
  epochs   = 50,        # 增加到50-100
  T        = 1000,
  lr       = 1e-4,      # 降低学习率（UNet参数多）
  schedule = beta_cosine(T = 1000),  # 余弦schedule更稳定
  verbose  = TRUE,
  seed     = 42,
  use_unet = TRUE       # 默认TRUE，显式指定
)
```

### 2. 对比：使用简单CNN

```r
# 回到原始CNN（仅用于对比）
fit_img_simple <- train_diffusion_image(
  train_dl = train_dl,
  epochs   = 50,
  T        = 1000,
  lr       = 5e-4,
  schedule = beta_linear(T = 1000),
  verbose  = TRUE,
  seed     = 42,
  use_unet = FALSE      # 使用简单CNN
)
```

### 3. 采样

```r
# 生成16张样本
samples <- sample_ddpm(
  fit       = fit_img,
  n         = 16L,
  steps     = fit_img$T,
  shape_img = c(28, 28)
)

# 可视化
par(mfrow = c(4, 4), mar = c(0.1, 0.1, 0.1, 0.1))
for (i in 1:16) {
  img <- samples[i, , ]
  image(1:28, 1:28, t(apply(img, 2, rev)),
        col = gray.colors(256), axes = FALSE)
}
```

---

## 优化技巧

### 训练阶段优化

#### 1. **学习率调度**
```r
# 推荐使用Cosine Schedule而非Linear
schedule <- beta_cosine(T = 1000)

# 配合降低初始学习率
lr <- 1e-4  # UNet用1e-4，CNN用5e-4
```

#### 2. **批量大小**
```r
# 增大批量大小提高稳定性
batch_size <- 256  # 从128改为256
```

#### 3. **梯度剪裁（已内置）**
- 防止梯度爆炸
- 特别重要：UNet有更多层

#### 4. **早停（推荐）**
```r
# 监控validation loss，避免过拟合
# 建议在50-100 epochs时观察loss曲线
```

### 采样阶段优化

#### 1. **采样步数**
```r
# 更多步数 = 更好质量但更慢
samples_fast <- sample_ddpm(fit, n = 100, steps = 50)   # 快速，质量一般
samples_good <- sample_ddpm(fit, n = 100, steps = 500)  # 平衡
samples_best <- sample_ddpm(fit, n = 100, steps = 1000) # 最优质量
```

#### 2. **采样技巧**
```r
# 1. 使用不同的初始噪声扩展多样性
# 2. 温度控制（若支持）
# 3. 条件生成（若模型支持）
```

---

## 预期改进

### 训练Loss曲线

```
简单CNN:     ——————\————————
             逐步下降但缓慢

UNet:        ———\———\———\———
             快速下降，更稳定
```

### 生成质量对比

**简单CNN生成**：
- 噪声多，细节模糊
- 数字笔画断断续续
- 整体结构不清晰

**UNet生成**：
- 数字轮廓清晰
- 笔画连贯自然
- 多样性更好

---

## 参数调优建议

### 小规模实验（5个epoch快速验证）

```r
fit_test <- train_diffusion_image(
  train_dl = train_dl,
  epochs   = 5,
  T        = 1000,
  lr       = 1e-4,
  schedule = beta_cosine(T = 1000),
  verbose  = TRUE,
  seed     = 42,
  use_unet = TRUE
)
```

### 标准训练（50-100个epoch）

```r
fit_img <- train_diffusion_image(
  train_dl = train_dl,
  epochs   = 100,        # 根据loss曲线调整
  T        = 1000,
  lr       = 1e-4,
  schedule = beta_cosine(T = 1000),
  verbose  = TRUE,
  seed     = 42,
  use_unet = TRUE
)

# 生成高质量样本
samples <- sample_ddpm(
  fit       = fit_img,
  n         = 1000L,
  steps     = fit_img$T,  # 全步数采样
  shape_img = c(28, 28)
)
```

### 超参数搜索范围

| 参数 | 简单CNN | UNet |
|------|--------|------|
| lr | 5e-4 ~ 1e-3 | 5e-5 ~ 1e-4 |
| batch_size | 128 | 256 |
| epochs | 50-100 | 100-200 |
| t_dim | 16 | 128 |
| ch (通道数) | 32-64 | 64 |
| ch_mult | - | [1,2,4] |

---

## 故障排除

### 问题1：训练速度慢
**原因**：UNet参数多  
**解决**：
- 减少ch_mult：`c(1,2)` 而非 `c(1,2,4)`
- 减少num_res_blocks：从2改为1
- 使用更小的t_dim：128改为64

### 问题2：Loss不下降
**原因**：学习率太高，梯度爆炸  
**解决**：
- 降低lr：1e-4 → 5e-5
- 检查梯度剪裁是否生效
- 检查数据是否正确归一化

### 问题3：生成质量仍差
**原因**：训练不足或模型太小  
**解决**：
- 增加epochs到200+
- 增加ch：64 → 128
- 使用更复杂的schedule

---

## 下一步改进方向

1. **条件扩散**：加入label条件生成特定数字
2. **Attention优化**：使用Flash-Attention加速
3. **多尺度损失**：在不同resolution添加辅助loss
4. **蒸馏**：训练小模型模仿大模型
5. **离散扩散**：用于纯离散数据

---

## 参考

- DDPM原论文：Ho et al. (2020)
- U-Net架构：Ronneberger et al. (2015)
- Diffusion Models实现细节：Nichol & Dhariwal (2021)
