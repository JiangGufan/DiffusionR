library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/src/kernels.cpp")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/datasets.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/metrics.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/model_cnn.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/model_mlp.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/parallel.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/sampling.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/schedules.R")
source("/Users/jiang/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/R/trainers.R")



# ==== 0. 加载必要包 ====
library(torch)
library(ggplot2)
library(coro)



# 1. 数据做一个简单标准化（可逆的）
X <- swiss_roll(n = 2000, noise = 0.1, seed = 1)
Xmean <- colMeans(X)
Xsd   <- apply(X, 2, sd)
X_std <- scale(X, center = Xmean, scale = Xsd)

# 2. 用 cosine schedule + 稍微多一点 epoch
sch <- beta_cosine(T = 1000)

fit_dist <- train_diffusion_dist(
  X         = X_std,
  epochs    = 300,        # 可以先用 150 看趋势
  T         = 1000,
  lr        = 5e-4,
  batch_size = 512,
  schedule  = sch,
  verbose   = TRUE,
  seed      = 42
)

# 3. 采样
fake_std <- sample_ddpm(
  fit   = fit_dist,
  n     = 2000,
  steps = fit_dist$T
)

# 4. 把样本还原回原坐标
fake <- sweep(fake_std, 2, Xsd, `*`)
fake <- sweep(fake,     2, Xmean, `+`)
colnames(fake) <- c("x1", "x2")

# 5. 画图
plot_2d_samples(real = X, fake = fake)





# ==== 1. 生成 Swiss roll 数据 ====
set.seed(1)
real <- swiss_roll(n = 2000, noise = 0.1, seed = 1)  # [2000,2]

# 简单看一下原始分布
qplot(real[,1], real[,2]) + coord_equal() + theme_minimal()

# ==== 2. 训练 diffusion MLP（DDPM 噪声预测） ====
fit_dist <- train_diffusion_dist(
  X         = real,
  epochs    = 100,      # 可以先用 20 快速试一下，再拉到 100+
  T         = 1000,
  lr        = 1e-3,
  batch_size = 256,
  schedule  = beta_linear(T = 1000),
  verbose   = TRUE,
  seed      = 42
)
# 返回 list(model, schedule, T, type="ddpm_mlp", dim=2)

# ==== 3. 用 DDPM 反向采样生成样本 ====
fake <- sample_ddpm(
  fit   = fit_dist,
  n     = 2000,       # 生成同数量样本
  steps = fit_dist$T  # 全 T 步
)

# 给 fake 补上列名，方便 plot_2d_samples 使用
colnames(fake) <- c("x1", "x2")

# ==== 4. 画真实/生成对比散点图 ====
p <- plot_2d_samples(real = real, fake = fake)
print(p)





library(torch)
library(torchvision)
library(coro)

# # 第一次需要：
# remotes::install_github("mlverse/torchvision")

train_dl <- mnist_train_dataloader(batch_size = 128)
it <- train_dl$.iter()
b  <- it$.next()
img <- b[[1]]
img$size()


# 训练一个小模型试试，先 1~3 个 epoch
fit_img <- train_diffusion_image(
  train_dl = train_dl,
  epochs   = 1,         # 可以先 1 试通，再加
  T        = 1000,
  lr       = 2e-4,
  schedule = beta_linear(T = 1000),
  verbose  = TRUE,
  seed     = 42
)
# 返回 list(model, schedule, T, type="ddpm_cnn")


# 生成 16 张 28x28 的图片
samples <- sample_ddpm(
  fit       = fit_img,
  n         = 16L,
  steps     = fit_img$T,         # 全 T 步
  shape_img = c(28, 28)
)

# 'samples'：预期是 [16, 28, 28] 的数组，范围在 [0,1]
dim(samples)
range(samples)

par(mfrow = c(4,4), mar = c(0.1,0.1,0.1,0.1))

for(i in 1:16){
  img <- samples[i,,]                # [28,28]
  # image 默认 (0,1) 是白到黑，我们翻转一下 y 轴：
  image(
    1:28, 1:28, t(apply(img, 2, rev)),
    col = gray.colors(256),
    axes = FALSE
  )
}

