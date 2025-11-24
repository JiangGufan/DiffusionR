# Minimal CNN epsilon-predictor for 28x28 grayscale images
# build_cnn_eps <- function(ch = 32){
#   torch::nn_module(
#     classname = "cnn_eps",
#     initialize = function(){
#       self$emb <- torch::nn_sequential(
#         torch::nn_linear(1, ch), torch::nn_silu(),
#         torch::nn_linear(ch, ch)
#       )
#       self$conv1 <- torch::nn_conv2d(2, ch, kernel_size = 3, padding = 1)
#       self$conv2 <- torch::nn_conv2d(ch, ch, kernel_size = 3, padding = 1)
#       self$conv3 <- torch::nn_conv2d(ch, 1, kernel_size = 3, padding = 1)
#     },
#     forward = function(x, t_frac){
#       B <- x$size()[1]; H <- x$size()[3]; W <- x$size()[4]
#       em <- self$emb(t_frac)$view(c(B,1,1,1))$expand(c(B,1,H,W))
#       x_in <- torch::torch_cat(list(x, em), dim = 2)
#       h <- torch::nnf_relu(self$conv1(x_in))
#       h <- torch::nnf_relu(self$conv2(h))
#       out <- self$conv3(h)
#       out
#     }
#   )
# }

## 一个跑的通的版本
# build_cnn_eps <- function(ch = 32){
#   torch::nn_module(
#     classname = "cnn_eps",
#     initialize = function(){
#       # 时间 embedding: 1 -> ch -> 1
#       self$emb <- torch::nn_sequential(
#         torch::nn_linear(1, ch), torch::nn_silu(),
#         torch::nn_linear(ch, 1L)
#       )
#       self$conv1 <- torch::nn_conv2d(2, ch, kernel_size = 3, padding = 1)
#       self$conv2 <- torch::nn_conv2d(ch, ch, kernel_size = 3, padding = 1)
#       self$conv3 <- torch::nn_conv2d(ch, 1, kernel_size = 3, padding = 1)
#     },
#     forward = function(x, t_frac){
#       B <- x$size()[1]; H <- x$size()[3]; W <- x$size()[4]
#       # t_frac 形状: [B] -> 需要转为 [B, 1] 然后通过 embedding
#       if (length(t_frac$shape) == 1) {
#         t_frac <- t_frac$unsqueeze(2)  # [B] -> [B, 1]
#       }
#       em <- self$emb(t_frac)  # [B, 1] -> [B, 1]
#       em <- em$view(c(B, 1, 1, 1))$expand(c(B, 1, H, W))  # [B, 1, H, W]
#       x_in <- torch::torch_cat(list(x, em), dim = 2)  # [B, 1+1, H, W] = [B, 2, H, W]
#       h <- torch::nnf_relu(self$conv1(x_in))
#       h <- torch::nnf_relu(self$conv2(h))
#       out <- self$conv3(h)
#       out
#     }
#   )
# }


# ===== 简单的残差块 =====
resnet_block <- function(in_ch, out_ch, t_emb_dim) {
  torch::nn_module(
    classname = "resnet_block",
    initialize = function() {
      self$in_ch  <- in_ch
      self$out_ch <- out_ch

      self$conv1 <- torch::nn_conv2d(in_ch, out_ch, kernel_size = 3, padding = 1)
      # ✅ norm1 的通道数 = in_ch
      self$norm1 <- torch::nn_group_norm(
        num_groups   = min(32L, in_ch),
        num_channels = in_ch
      )

      self$time_mlp <- torch::nn_sequential(
        torch::nn_silu(),
        torch::nn_linear(t_emb_dim, out_ch)
      )

      self$conv2 <- torch::nn_conv2d(out_ch, out_ch, kernel_size = 3, padding = 1)
      # ✅ norm2 的通道数 = out_ch
      self$norm2 <- torch::nn_group_norm(
        num_groups   = min(32L, out_ch),
        num_channels = out_ch
      )

      if (in_ch != out_ch) {
        self$res_proj <- torch::nn_conv2d(in_ch, out_ch, kernel_size = 1)
      } else {
        self$res_proj <- NULL
      }
    },

    forward = function(x, t_emb) {
      # x: [B, in_ch, H, W]
      res <- x
      B   <- x$size()[1]

      # 这里 x 通道数 = in_ch，对应上面的 norm1 设置
      h <- self$norm1(x)
      h <- torch::nnf_silu(h)
      h <- self$conv1(h)  # 现在通道变成 out_ch

      # 添加时间嵌入： [B, out_ch] -> [B, out_ch, 1, 1]
      t_add <- self$time_mlp(t_emb)$view(c(B, self$out_ch, 1L, 1L))
      h <- h + t_add

      h <- self$norm2(h)  # 这里输入通道 = out_ch
      h <- torch::nnf_silu(h)
      h <- self$conv2(h)

      if (!is.null(self$res_proj)) {
        res <- self$res_proj(res)
      }
      h + res
    }
  )
}


# ===== Attention 块（MNIST 这种小图像可选，但有助于提高质量） =====
attention_block <- function(ch) {
  torch::nn_module(
    classname = "attention_block",
    initialize = function() {
      self$norm <- torch::nn_group_norm(32, ch)
      self$q_proj <- torch::nn_conv2d(ch, ch, kernel_size = 1)
      self$k_proj <- torch::nn_conv2d(ch, ch, kernel_size = 1)
      self$v_proj <- torch::nn_conv2d(ch, ch, kernel_size = 1)
      self$out_proj <- torch::nn_conv2d(ch, ch, kernel_size = 1)
    },
    forward = function(x) {
      # x: [B, ch, H, W]
      B <- x$size()[1]; ch <- x$size()[2]; H <- x$size()[3]; W <- x$size()[4]
      
      res <- x
      x <- self$norm(x)
      
      # 计算 Q, K, V
      q <- self$q_proj(x)$view(c(B, ch, -1L))  # [B, ch, H*W]
      k <- self$k_proj(x)$view(c(B, ch, -1L))  # [B, ch, H*W]
      v <- self$v_proj(x)$view(c(B, ch, -1L))  # [B, ch, H*W]
      
      # 注意力：[B, H*W, ch] @ [B, ch, H*W] -> [B, H*W, H*W]
      q <- q$permute(c(1L, 3L, 2L))  # [B, H*W, ch]
      k <- k$permute(c(1L, 3L, 2L))  # [B, H*W, ch]
      v <- v$permute(c(1L, 3L, 2L))  # [B, H*W, ch]
      
      attn <- torch::torch_bmm(q, k$permute(c(1L, 3L, 2L)))  # [B, H*W, H*W]
      attn <- attn / sqrt(ch)
      attn <- torch::nnf_softmax(attn, dim = -1L)
      
      out <- torch::torch_bmm(attn, v)  # [B, H*W, ch]
      out <- out$permute(c(1L, 3L, 2L))$view(c(B, ch, H, W))  # [B, ch, H, W]
      
      out <- self$out_proj(out)
      out + res
    }
  )
}

# ===== 主要 UNet 架构用于 DDPM =====
build_unet_eps <- function(ch = 64, ch_mult = c(1, 2), t_dim = 128, num_res_blocks = 2) {
  torch::nn_module(
    classname = "unet_eps",

    initialize = function() {
      self$t_dim <- t_dim

      # ===== 时间 embedding MLP：t_dim → 4*t_dim =====
      self$time_embedding <- torch::nn_sequential(
        torch::nn_linear(t_dim, t_dim * 4),
        torch::nn_silu(),
        torch::nn_linear(t_dim * 4, t_dim * 4)
      )

      base_ch <- ch       # 64
      ch2     <- base_ch * 2  # 128

      # ===== 输入卷积：1×28×28 → 64×28×28 =====
      self$input_conv <- torch::nn_conv2d(1, base_ch, kernel_size = 3, padding = 1)

      # ===== Encoder level 1：分辨率 28×28，通道 64 =====
      self$down1_block1 <- resnet_block(base_ch, base_ch, t_dim * 4)()
      self$down1_block2 <- resnet_block(base_ch, base_ch, t_dim * 4)()

      # 下采样：64×28×28 → 128×14×14
      self$downsample1 <- torch::nn_conv2d(
        in_channels  = base_ch,
        out_channels = ch2,
        kernel_size  = 4,
        stride       = 2,
        padding      = 1
      )

      # ===== Encoder level 2：分辨率 14×14，通道 128 =====
      self$down2_block1 <- resnet_block(ch2, ch2, t_dim * 4)()
      self$down2_block2 <- resnet_block(ch2, ch2, t_dim * 4)()

      # ===== Bottleneck：14×14, 128 通道 =====
      self$mid_block1 <- resnet_block(ch2, ch2, t_dim * 4)()
      self$mid_block2 <- resnet_block(ch2, ch2, t_dim * 4)()

      # ===== Decoder level 2（仍然 14×14）：先跟 skip2 拼，再做两个 ResBlock =====
      # mid(128) + skip2(128) → cat 成 256 通道
      self$up_block2a <- resnet_block(in_ch = ch2 * 2, out_ch = ch2, t_dim * 4)()  # 256→128
      self$up_block2b <- resnet_block(in_ch = ch2,     out_ch = ch2, t_dim * 4)()  # 128→128

      # 上采样：128×14×14 → 64×28×28
      self$up_conv1 <- torch::nn_conv_transpose2d(
        in_channels  = ch2,
        out_channels = base_ch,
        kernel_size  = 4,
        stride       = 2,
        padding      = 1
      )

      # ===== Decoder level 1：分辨率 28×28 =====
      # up(64) + skip1(64) → 128
      self$up_block1a <- resnet_block(in_ch = base_ch * 2, out_ch = base_ch, t_dim * 4)() # 128→64
      self$up_block1b <- resnet_block(in_ch = base_ch,     out_ch = base_ch, t_dim * 4)() # 64→64

      # ===== 输出头：64×28×28 → 1×28×28 =====
      self$final_norm <- torch::nn_group_norm(
        num_groups   = min(32L, base_ch),
        num_channels = base_ch
      )
      self$final_conv <- torch::nn_conv2d(base_ch, 1, kernel_size = 3, padding = 1)
    },

    forward = function(x, t_frac) {
      # x: [B, 1, 28, 28], t_frac: [B] or [B,1]
      B <- x$size()[1]

      # ---- 处理 t_frac → [B,1] ----
      if (length(t_frac$shape) == 1) {
        t_frac <- t_frac$unsqueeze(2)  # [B] -> [B,1]
      }

      # ---- sinusoidal 时间编码 ----
      t_dim <- self$t_dim
      half  <- as.integer(t_dim / 2)

      freqs <- torch::torch_exp(
        torch::torch_arange(
          0, half - 1,
          dtype  = torch::torch_float(),
          device = t_frac$device
        ) * (-log(10000.0) / half)
      )
      args <- t_frac$matmul(freqs$view(c(1L, half)))
      t_emb <- torch::torch_cat(
        list(torch::torch_sin(args), torch::torch_cos(args)),
        dim = 2
      )  # [B, t_dim]

      # 通过时间 embedding MLP 得到 [B, 4*t_dim]
      t_emb <- self$time_embedding(t_emb)

      # ===== Encoder =====
      h <- self$input_conv(x)          # [B, 64, 28, 28]

      # level 1, 28×28
      h <- self$down1_block1(h, t_emb) # [B, 64, 28, 28]
      h <- self$down1_block2(h, t_emb) # [B, 64, 28, 28]
      skip1 <- h                       # 保存 skip1: [B,64,28,28]

      # 下采样到 14×14
      h <- self$downsample1(h)         # [B,128,14,14]

      # level 2, 14×14
      h <- self$down2_block1(h, t_emb) # [B,128,14,14]
      h <- self$down2_block2(h, t_emb) # [B,128,14,14]
      skip2 <- h                       # 保存 skip2: [B,128,14,14]

      # ===== Bottleneck =====
      h <- self$mid_block1(h, t_emb)   # [B,128,14,14]
      h <- self$mid_block2(h, t_emb)   # [B,128,14,14]

      # ===== Decoder level 2（14×14）=====
      # 拼接 skip2：128 + 128 → 256
      h <- torch::torch_cat(list(h, skip2), dim = 2)      # [B,256,14,14]
      h <- self$up_block2a(h, t_emb)                     # [B,128,14,14]
      h <- self$up_block2b(h, t_emb)                     # [B,128,14,14]

      # 上采样到 28×28：128→64
      h <- self$up_conv1(h)                              # [B,64,28,28]

      # ===== Decoder level 1（28×28）=====
      h <- torch::torch_cat(list(h, skip1), dim = 2)     # [B,128,28,28]
      h <- self$up_block1a(h, t_emb)                     # [B,64,28,28]
      h <- self$up_block1b(h, t_emb)                     # [B,64,28,28]

      # ===== 输出头 =====
      h <- self$final_norm(h)
      h <- torch::nnf_silu(h)
      out <- self$final_conv(h)                          # [B,1,28,28]

      out
    }
  )
}


# ===== 保留原简单 CNN 为备选 =====
build_cnn_eps <- function(ch = 64, t_dim = 16) {
  torch::nn_module(
    classname = "cnn_eps",

    initialize = function() {
      self$t_dim <- t_dim

      self$time_mlp <- torch::nn_sequential(
        torch::nn_linear(t_dim, ch),
        torch::nn_silu(),
        torch::nn_linear(ch, ch)
      )

      self$conv1 <- torch::nn_conv2d(1,    ch, kernel_size = 3, padding = 1)
      self$conv2 <- torch::nn_conv2d(ch,   ch, kernel_size = 3, padding = 1)
      self$conv3 <- torch::nn_conv2d(ch,   ch, kernel_size = 3, padding = 1)
      self$conv4 <- torch::nn_conv2d(ch,   1,  kernel_size = 3, padding = 1)
    },

    forward = function(x, t_frac) {
      B <- x$size()[1]
      H <- x$size()[3]
      W <- x$size()[4]

      if (length(t_frac$shape) == 1) {
        t_frac <- t_frac$unsqueeze(2)
      }

      t_dim <- self$t_dim
      half  <- as.integer(t_dim/2)

      freqs <- torch::torch_exp(
        torch::torch_arange(
          0, half - 1,
          dtype  = torch::torch_float(),
          device = t_frac$device
        ) * (-log(10000.0)/half)
      )
      args <- t_frac$matmul(freqs$view(c(1, half)))
      t_emb <- torch::torch_cat(
        list(torch::torch_sin(args), torch::torch_cos(args)),
        dim = 2
      )

      t_h <- self$time_mlp(t_emb)
      t_h <- t_h$view(c(B, ch, 1, 1))

      h <- self$conv1(x)
      h <- h + t_h
      h <- torch::nnf_relu(h)

      h <- torch::nnf_relu(self$conv2(h))
      h <- torch::nnf_relu(self$conv3(h))
      out <- self$conv4(h)

      out
    }
  )
}
