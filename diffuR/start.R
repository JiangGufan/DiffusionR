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
  epochs    = 200,        # 可以先用 150 看趋势
  T         = 1000,
  lr        = 1e-4,
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

## ===== 1. 选择想展示的时间步（扩散方向：从干净数据到高噪声） =====
t_show <- c(0, 100, 200, 500, 1000)   # 可以自己改，比如多/少几个

## ===== 2. 构造各个时间步下的 "noised" 样本（在标准化空间） =====
snap_std <- list()
snap_std[["t0"]] <- as.matrix(X_std)   # t = 0：原始标准化数据

set.seed(123)  # 为了可重复
for (t in t_show[-1]) {
  snap_std[[paste0("t", t)]] <- q_sample_xt_given_x0(
    X0 = as.matrix(X_std),
    sa = sch$sqrt_alpha_bar[t],          # sqrt(alpha_bar_t)
    om = sch$sqrt_one_minus_alpha_bar[t] # sqrt(1 - alpha_bar_t)
  )
}

## ===== 3. 统一裁剪半径，避免极端 outlier 干扰可视化 =====
# 用训练数据本身来设定一个“合理半径”，比如 99% 分位
r2_real <- rowSums(X_std^2)
R <- sqrt(quantile(r2_real, 0.99))   # 你也可以改成 0.95、0.999 等

## 把真实数据先还原好（后面每个面板复用）
X_real_clip_std <- X_std[r2_real <= R^2, , drop = FALSE]
X_real_clip <- sweep(X_real_clip_std, 2, Xsd,   `*`)
X_real_clip <- sweep(X_real_clip,     2, Xmean, `+`)
colnames(X_real_clip) <- c("x1", "x2")

## ===== 4. 组装成一个大 data.frame，用 facet 展示每个 t =====
df_list <- list()

for (nm in names(snap_std)) {
  Xt_std <- snap_std[[nm]]
  
  # 裁剪掉半径太大的点（和真实数据用同一个 R，方便比较）
  r2 <- rowSums(Xt_std^2)
  keep <- r2 <= R^2
  Xt_std_clip <- Xt_std[keep, , drop = FALSE]
  
  # 还原回原坐标
  Xt <- sweep(Xt_std_clip, 2, Xsd,   `*`)
  Xt <- sweep(Xt,          2, Xmean, `+`)
  colnames(Xt) <- c("x1", "x2")
  
  step_lab <- paste0("t = ", sub("t", "", nm))
  
  df_fake <- data.frame(
    x1   = Xt[,1],
    x2   = Xt[,2],
    type = "noised",
    step = step_lab
  )
  
  df_real <- data.frame(
    x1   = X_real_clip[,1],
    x2   = X_real_clip[,2],
    type = "real",
    step = step_lab
  )
  
  df_list[[length(df_list) + 1]] <- rbind(df_real, df_fake)
}

df_all <- do.call(rbind, df_list)


##  保证 step 的顺序是 t = 0, 25, 100, 500, 1000
t_show <- c(0, 100, 200, 500, 1000)
df_all$step <- factor(df_all$step,
                      levels = paste0("t = ", t_show))
ggplot(df_all, aes(x = x1, y = x2, color = type)) +
  geom_point(alpha = 0.45, size = 0.6) +
  coord_equal() +
  facet_grid(. ~ step) +   # 横向排成一行
  # 或者 facet_wrap(~ step, nrow = 1) 也可以
  scale_color_manual(values = c("real" = "#00BFC4", "noised" = "#F8766D")) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "x1",
    y = "x2",
    color = "type",
    title = "Forward diffusion of swiss roll under cosine schedule"
  )




# # # 4. 把样本还原回原坐标
# # fake <- sweep(fake_std, 2, Xsd, `*`)
# # fake <- sweep(fake,     2, Xmean, `+`)
# # colnames(fake) <- c("x1", "x2")
# # 
# # # 5. 画图
# # plot_2d_samples(real = X, fake = fake)
# 
# ## 3.5 在标准化空间裁剪：只保留半径 <= R 的点
# R <- sqrt(quantile(r2, 0.99))  # 保留 99% 最近的点
# r2 <- rowSums(fake_std^2)            # 每个点到原点的平方距离
# keep <- r2 <= R^2                    # 逻辑向量
# 
# fake_std_clip <- fake_std[keep, , drop = FALSE]
# 
# ## 4. 把裁剪后的样本还原回原坐标
# fake_clip <- sweep(fake_std_clip, 2, Xsd,   `*`)
# fake_clip <- sweep(fake_clip,     2, Xmean, `+`)
# colnames(fake_clip) <- c("x1", "x2")
# 
# ## 5. 画图
# plot_2d_samples(real = X, fake = fake_clip)




## ===== 0. 生成 8-Gaussians ring 数据 =====
X <- gauss8_ring(n = 2000, radius = 2, noise = 0.1, seed = 1)

# 1. 标准化（可逆）
Xmean <- colMeans(X)
Xsd   <- apply(X, 2, sd)
X_std <- scale(X, center = Xmean, scale = Xsd)

# 2. cosine schedule
sch <- beta_cosine(T = 1000)


fit_dist <- train_diffusion_dist(
  X         = X_std,
  epochs    = 200,        # 可以先用 150 看趋势
  T         = 1000,
  lr        = 1e-4,
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


# 3. 想展示的时间步（从干净到高噪声）
t_show <- c(0, 100, 200, 500, 1000)

## ===== 3. 构造每个 t 下的 noised 样本（在标准化空间） =====
snap_std <- list()
snap_std[["t0"]] <- as.matrix(X_std)

set.seed(123)
for (t in t_show[-1]) {
  snap_std[[paste0("t", t)]] <- q_sample_xt_given_x0(
    X0 = as.matrix(X_std),
    sa = sch$sqrt_alpha_bar[t],
    om = sch$sqrt_one_minus_alpha_bar[t]
  )
}

## ===== 4. 根据真实数据半径裁剪，避免极端 outlier =====
r2_real <- rowSums(X_std^2)
R <- sqrt(quantile(r2_real, 0.99))

X_real_clip_std <- X_std[r2_real <= R^2, , drop = FALSE]
X_real_clip <- sweep(X_real_clip_std, 2, Xsd,   `*`)
X_real_clip <- sweep(X_real_clip,     2, Xmean, `+`)
colnames(X_real_clip) <- c("x1", "x2")

## ===== 5. 组装成 df_all，用 facet_grid 横向展示 =====
df_list <- list()

for (nm in names(snap_std)) {
  Xt_std <- snap_std[[nm]]
  
  # 用同一个半径 R 裁剪
  r2 <- rowSums(Xt_std^2)
  keep <- r2 <= R^2
  Xt_std_clip <- Xt_std[keep, , drop = FALSE]
  
  # 还原到原坐标
  Xt <- sweep(Xt_std_clip, 2, Xsd,   `*`)
  Xt <- sweep(Xt,          2, Xmean, `+`)
  colnames(Xt) <- c("x1", "x2")
  
  step_lab <- paste0("t = ", sub("t", "", nm))
  
  df_fake <- data.frame(
    x1   = Xt[, 1],
    x2   = Xt[, 2],
    type = "noised",
    step = step_lab
  )
  df_real <- data.frame(
    x1   = X_real_clip[, 1],
    x2   = X_real_clip[, 2],
    type = "real",
    step = step_lab
  )
  
  df_list[[length(df_list) + 1]] <- rbind(df_real, df_fake)
}

df_all <- do.call(rbind, df_list)

# 控制 facet 顺序：t = 0, 50, 200, 500, 1000
df_all$step <- factor(
  df_all$step,
  levels = paste0("t = ", t_show)
)

## ===== 6. 画图 =====
ggplot(df_all, aes(x = x1, y = x2, color = type)) +
  geom_point(alpha = 0.45, size = 0.6) +
  coord_equal() +
  facet_grid(. ~ step) +
  scale_color_manual(values = c("real" = "#00BFC4", "noised" = "#F8766D")) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position    = "right",
    panel.grid.minor   = element_blank()
  ) +
  labs(
    x = "x1",
    y = "x2",
    color = "type",
    title = "Forward diffusion of 8-Gaussians ring under cosine schedule"
  )






## ===== 0. 生成 heart 数据 =====
X <- heart2d(n = 2000, noise = 0.08, seed = 1)

# 1. 标准化（可逆）
Xmean <- colMeans(X)
Xsd   <- apply(X, 2, sd)
X_std <- scale(X, center = Xmean, scale = Xsd)

# 2. cosine schedule
sch <- beta_cosine(T = 1000)


fit_dist <- train_diffusion_dist(
  X         = X_std,
  epochs    = 200,        # 可以先用 150 看趋势
  T         = 1000,
  lr        = 1e-4,
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


# 3. 想展示的时间步（从干净到高噪声）
t_show <- c(0, 100, 200, 500, 1000)

## ===== 3. 构造每个 t 下的 noised 样本（在标准化空间） =====
snap_std <- list()
snap_std[["t0"]] <- as.matrix(X_std)

set.seed(123)
for (t in t_show[-1]) {
  snap_std[[paste0("t", t)]] <- q_sample_xt_given_x0(
    X0 = as.matrix(X_std),
    sa = sch$sqrt_alpha_bar[t],
    om = sch$sqrt_one_minus_alpha_bar[t]
  )
}

## ===== 4. 根据真实数据半径裁剪，避免极端 outlier =====
r2_real <- rowSums(X_std^2)
R <- sqrt(quantile(r2_real, 0.99))

X_real_clip_std <- X_std[r2_real <= R^2, , drop = FALSE]
X_real_clip <- sweep(X_real_clip_std, 2, Xsd,   `*`)
X_real_clip <- sweep(X_real_clip,     2, Xmean, `+`)
colnames(X_real_clip) <- c("x1", "x2")

## ===== 5. 组装成 df_all，用 facet_grid 横向展示 =====
df_list <- list()

for (nm in names(snap_std)) {
  Xt_std <- snap_std[[nm]]
  
  # 用同一个半径 R 裁剪
  r2 <- rowSums(Xt_std^2)
  keep <- r2 <= R^2
  Xt_std_clip <- Xt_std[keep, , drop = FALSE]
  
  # 还原到原坐标
  Xt <- sweep(Xt_std_clip, 2, Xsd,   `*`)
  Xt <- sweep(Xt,          2, Xmean, `+`)
  colnames(Xt) <- c("x1", "x2")
  
  step_lab <- paste0("t = ", sub("t", "", nm))
  
  df_fake <- data.frame(
    x1   = Xt[, 1],
    x2   = Xt[, 2],
    type = "noised",
    step = step_lab
  )
  df_real <- data.frame(
    x1   = X_real_clip[, 1],
    x2   = X_real_clip[, 2],
    type = "real",
    step = step_lab
  )
  
  df_list[[length(df_list) + 1]] <- rbind(df_real, df_fake)
}

df_all <- do.call(rbind, df_list)

# 控制 facet 顺序：t = 0, 50, 200, 500, 1000
df_all$step <- factor(
  df_all$step,
  levels = paste0("t = ", t_show)
)

## ===== 6. 画图 =====
ggplot(df_all, aes(x = x1, y = x2, color = type)) +
  geom_point(alpha = 0.45, size = 0.6) +
  coord_equal() +
  facet_grid(. ~ step) +
  scale_color_manual(values = c("real" = "#00BFC4", "noised" = "#F8766D")) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position    = "right",
    panel.grid.minor   = element_blank()
  ) +
  labs(
    x = "x1",
    y = "x2",
    color = "type",
    title = "Forward diffusion of Heart under cosine schedule"
  )





library(torch)
# ===== 0. 一开始就设置线程 =====
torch_set_num_threads(8)
torch_set_num_interop_threads(2)

library(torchvision)
library(coro)

cat("intra-op threads:", torch_get_num_threads(), "\n")
cat("inter-op threads:", torch_get_num_interop_threads(), "\n")


# # 第一次需要：
# remotes::install_github("mlverse/torchvision")

# 大 batch + 多 worker
train_dl <- mnist_train_dataloader(
  batch_size  = 512,   # 比原来的 128 大不少
  num_workers = 4L     # 后台预取 batch
)
it <- train_dl$.iter()
b  <- it$.next()
img <- b[[1]]
img$size()


# ========== 步骤 2: 定义调度 ==========
# Cosine schedule 比线性schedule更稳定
schedule_cosine <- beta_cosine(T = 1000)
schedule_linear <- beta_linear(T = 1000)


# fit_unet_quick <- train_diffusion_image(
#   train_dl = train_dl,
#   epochs   = 5,
#   T        = 1000,
#   lr       = 1e-4,
#   schedule = schedule_cosine,
#   verbose  = TRUE,
#   seed     = 42,
#   use_unet = TRUE
# )

# 训练一个小模型试试，先 1~3 个 epoch
fit_img <- train_diffusion_image(
  train_dl = train_dl,
  epochs   = 10,         # 可以先 1 试通，再加
  T        = 1000,
  lr       = 1e-4,
  schedule = schedule_cosine,   # 此前为beta_linear(T = 1000),
  verbose  = TRUE,
  seed     = 42
)
# 返回 list(model, schedule, T, type="ddpm_cnn")

# 先用小 steps 测一下
samples_small <- sample_ddpm(
  fit       = fit_img,
  n         = 2L,
  steps     = 50L,         # 先 50 步
  shape_img = c(28, 28)
)

dim(samples_small)
range(samples_small)

# 画 2 张图
par(mfrow = c(1, 2), mar = c(0.1, 0.1, 0.1, 0.1))

for (i in 1:2) {
  img <- samples_small[i, , ]  # [28,28]
  
  # 翻转一下 y 轴，避免倒着的数字
  image(
    1:28, 1:28,
    t(apply(img, 2, rev)),
    col  = gray.colors(256),
    axes = FALSE
  )
}


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


# 假设你已经有：
# fit_img <- train_diffusion_image(..., T = 1000, ...)

# 想看的时间步（从极度马赛克 → 清晰数字）
t_vis <- c(1000, 500, 200, 100, 0)

# 采样一条轨迹（这里 n=16，同时采 16 个样本，其实我们下面只画第 1 个）
traj <- sample_ddpm_trajectory(
  fit       = fit_img,
  n         = 16L,
  t_show    = t_vis,
  shape_img = c(28, 28),
  seed      = 123
)

# 画图：一行 5 张图，展示同一个样本在不同 t 的状态
par(mfrow = c(4, length(t_vis)), mar = c(0.1, 0.1, 2, 0.1))

for (j in seq_along(t_vis)) {
  t  <- t_vis[j]
  Im <- traj[[j]]  # [n,28,28]
  
  # 这里选第一个样本的轨迹：Im[1,,]
  img <- Im[10, , ]
  
  image(
    1:28, 1:28, t(apply(img, 2, rev)),
    col  = gray.colors(256),
    axes = FALSE,
    main = paste0("t = ", t)
  )
}


par(mfrow = c(length(t_vis), 4), mar = c(0.1, 0.1, 0.1, 0.1))

for (j in seq_along(t_vis)) {
  Im <- traj[[j]]
  for (i in 1:4) {   # 每个 t 只展示 4 个样本，防止太挤
    img <- Im[i, , ]
    image(
      1:28, 1:28, t(apply(img, 2, rev)),
      col  = gray.colors(256),
      axes = FALSE
    )
  }
}


