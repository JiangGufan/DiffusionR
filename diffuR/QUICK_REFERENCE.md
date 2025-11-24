# UNet vs CNN 快速参考

## 核心改进总结

### 1️⃣ 架构改进

**简单CNN（原始）：**
```
Input → Conv → Conv → Conv → Conv → Output
        缺少残差连接，信息流动差
```

**UNet（推荐）：**
```
         ↓Encoder        Decoder↑
Input → Conv ← ────────── Conv → Output
         ↓    ↓ResBlock↑ ↓
        Conv ← ──Att──  → Conv
         ↓    ↓ResBlock↑ ↓
        Conv ← ─Bottle─ → Conv
        
核心特性：
✓ 残差块：缓解梯度消失
✓ 跳连接：多尺度信息融合
✓ 注意力：学习长程依赖
✓ GroupNorm：更稳定
```

### 2️⃣ 参数量对比

| 指标 | 简单CNN | UNet |
|------|--------|------|
| 参数数 | ~200K | ~5M |
| 内存 | 低 | 中 |
| 速度 | 快 | 中等 |
| 生成质量 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### 3️⃣ 快速上手

**最少改动方案：**
```r
# 原来的代码
fit_img <- train_diffusion_image(train_dl = train_dl)

# 改为（只需加一个参数）
fit_img <- train_diffusion_image(
  train_dl = train_dl,
  epochs = 100,
  lr = 1e-4,
  schedule = beta_cosine(T = 1000),
  use_unet = TRUE        # ← 唯一新增
)
```

### 4️⃣ 关键参数

**学习率（最重要）：**
- UNet: `1e-4` ← 降低（参数多）
- CNN: `5e-4`

**Schedule（重要）：**
- UNet: `beta_cosine()` ← 推荐
- CNN: `beta_linear()`

**批量大小：**
- 推荐: `256`（相对于原来的128提高稳定性）

**Epochs：**
- 快速验证: `5`（确认能跑）
- 标准训练: `50-100`
- 最优结果: `200+`

### 5️⃣ 预期结果改进

| 方面 | 简单CNN | UNet |
|------|--------|------|
| Loss收敛 | 缓慢 | **快速** |
| 数字清晰度 | 模糊 | **清晰** |
| 笔画连贯性 | 断断续续 | **自然** |
| 多样性 | 低 | **高** |
| 训练稳定性 | 一般 | **优秀** |

### 6️⃣ 故障排除

**问题：Loss不下降**
→ 降低lr到1e-5，或检查数据归一化

**问题：显存爆炸**
→ 减少ch（64→32），或batch_size（256→128）

**问题：生成质量差**
→ 增加epochs（5→50→100），确保充分训练

### 7️⃣ 模型文件结构

```r
# 创建和使用自定义模型
model_custom <- build_unet_eps(
  ch = 64,           # 基础通道数（增大提高质量）
  ch_mult = c(1,2),  # 通道倍增（c(1,2,4)质量更好但慢）
  t_dim = 128,       # 时间嵌入维度
  num_res_blocks = 2 # 每层的残差块数
)
```

### 8️⃣ 采样质量控制

```r
# 快速采样（质量一般）
sample_ddpm(fit, n=100, steps=50)

# 平衡采样
sample_ddpm(fit, n=100, steps=500)

# 高质量采样（最慢）
sample_ddpm(fit, n=100, steps=1000)
```

---

## 实际应用流程

### Phase 1: 快速验证 (5分钟)
```r
fit <- train_diffusion_image(train_dl, epochs=5, use_unet=TRUE)
# 检查是否能正常运行，loss是否下降
```

### Phase 2: 标准训练 (30-60分钟)
```r
fit <- train_diffusion_image(train_dl, epochs=100, use_unet=TRUE)
# 等待收敛，观察loss曲线
```

### Phase 3: 高质量采样 (取决于步数)
```r
samples <- sample_ddpm(fit, n=1000, steps=1000, shape_img=c(28,28))
# 生成最终结果
```

---

## 为什么UNet更好？

1. **多尺度特征**
   - 编码器逐步下采样，捕捉全局结构
   - 解码器逐步上采样，恢复细节
   - 跳连接融合多尺度信息

2. **梯度流通**
   - 残差块允许梯度直接通过
   - 跳连接提供替代路径
   - 更稳定的训练动力学

3. **注意力机制**
   - 学习像素间关系（笔画的连贯性）
   - 特别重要：MNIST数字需要长程依赖

4. **设计验证**
   - UNet在医学影像、语义分割等任务上都SOTA
   - DDPM原论文核心架构就是UNet

---

## 一行命令总结

**从CNN升级到UNet：**
```r
# 原来
fit <- train_diffusion_image(train_dl)

# 现在（+ 3个关键参数）
fit <- train_diffusion_image(train_dl, 
                             epochs=100, 
                             lr=1e-4, 
                             use_unet=TRUE)
```

**效果：生成质量 ⭐⭐⭐⭐⭐ vs ⭐⭐**
