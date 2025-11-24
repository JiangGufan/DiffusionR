# Simple MLP epsilon-predictor for low-dimensional data
# build_mlp_eps <- function(dim_x, hidden = 256){
#   torch::nn_module(
#     classname = "mlp_eps",
#     initialize = function(){
#       self$net <- torch::nn_sequential(
#         torch::nn_linear(dim_x + 1L, hidden), torch::nn_silu(),
#         torch::nn_linear(hidden, hidden),     torch::nn_silu(),
#         torch::nn_linear(hidden, dim_x)
#       )
#     },
#     forward = function(x_t, t_frac){
#       self$net(torch::torch_cat(list(x_t, t_frac), dim = 2))
#     }
#   )
# }

# build_mlp_eps <- function(dim_x, hidden = 256, t_dim = 16){
#   torch::nn_module(
#     classname = "mlp_eps",
    
#     initialize = function(){
#       self$t_dim <- t_dim
      
#       # 时间嵌入先过一层 MLP
#       self$time_mlp <- torch::nn_sequential(
#         torch::nn_linear(t_dim, hidden),
#         torch::nn_silu()
#       )
      
#       # 主网络：输入 = x_t (dim_x) + time_hidden (hidden)
#       self$net <- torch::nn_sequential(
#         torch::nn_linear(dim_x + hidden, hidden), torch::nn_silu(),
#         torch::nn_linear(hidden, hidden),        torch::nn_silu(),
#         torch::nn_linear(hidden, hidden),        torch::nn_silu(),
#         torch::nn_linear(hidden, dim_x)
#       )
#     },
    
#     forward = function(x_t, t_frac){
#       # x_t: [B, dim_x], t_frac: [B,1]
#       B <- x_t$size()[1]
#       t_dim <- self$t_dim
#       half <- as.integer(t_dim/2)

#       # ===== sinusoidal 时间编码 =====
#       # freqs: [half]
#       freqs <- torch::torch_exp(
#         torch::torch_arange(
#           0, half - 1,
#           dtype = torch::torch_float(),
#           device = t_frac$device
#         ) * (-log(10000.0)/half)
#       )
#       # t_frac: [B,1], freqs: [1,half] -> args: [B,half]
#       args <- t_frac$matmul(freqs$view(c(1, half)))
#       t_emb <- torch::torch_cat(
#         list(torch::torch_sin(args), torch::torch_cos(args)),
#         dim = 2
#       ) # [B, t_dim]

#       # ===== 时间嵌入通过一层 MLP =====
#       t_h <- self$time_mlp(t_emb)  # [B, hidden]

#       # 拼到 x_t 后面
#       h_in <- torch::torch_cat(list(x_t, t_h), dim = 2)  # [B, dim_x + hidden]
#       self$net(h_in)  # 输出 [B, dim_x]，即 eps_pred
#     }
#   )
# }


# ===== 残差 MLP Block：对向量特征用的 =====
res_mlp_block <- function(hidden_dim, t_hidden_dim) {
  torch::nn_module(
    classname = "res_mlp_block",
    
    initialize = function() {
      self$hidden_dim     <- hidden_dim
      self$t_hidden_dim   <- t_hidden_dim
      
      # 主干层
      self$lin1  <- torch::nn_linear(hidden_dim, hidden_dim)
      self$lin2  <- torch::nn_linear(hidden_dim, hidden_dim)
      self$norm1 <- torch::nn_layer_norm(hidden_dim)
      self$norm2 <- torch::nn_layer_norm(hidden_dim)
      
      # 时间调制：把 time_hidden 投到 hidden_dim，作为一个“偏置”
      self$time_proj <- torch::nn_linear(t_hidden_dim, hidden_dim)
    },
    
    # h: [B, hidden_dim], t_h: [B, t_hidden_dim]
    forward = function(h, t_h) {
      res <- h
      
      # time embedding 投到与 h 相同的通道数
      t_add <- self$time_proj(t_h)  # [B, hidden_dim]
      
      x <- self$norm1(h)
      x <- torch::nnf_silu(x)
      x <- x + t_add                 # 每一层都注入时间信息
      x <- self$lin1(x)
      
      x <- self$norm2(x)
      x <- torch::nnf_silu(x)
      x <- self$lin2(x)
      
      x + res                        # 残差连接
    }
  )
}

# ===== 改进版 epsilon-MLP，用于向量分布（swiss roll 等） =====
build_mlp_eps <- function(dim_x, hidden = 256, t_dim = 64, n_blocks = 4) {
  torch::nn_module(
    classname = "mlp_eps",
    
    initialize = function() {
      self$t_dim <- t_dim
      self$hidden <- hidden
      
      # 1) 时间 sinusoidal embedding -> t_dim
      # 2) 再通过一层 MLP -> time_hidden (和 hidden 维度相同)
      self$time_mlp <- torch::nn_sequential(
        torch::nn_linear(t_dim, hidden),
        torch::nn_silu(),
        torch::nn_linear(hidden, hidden)   # 输出仍然是 [B, hidden]
      )
      
      # 输入投影：把 x_t 从 dim_x 投到 hidden 维度
      self$in_proj <- torch::nn_linear(dim_x, hidden)
      
      # 堆叠若干个残差 MLP block
      self$blocks <- torch::nn_module_list()
      for (i in seq_len(n_blocks)) {
        self$blocks$append(
          res_mlp_block(hidden_dim   = hidden,
                        t_hidden_dim = hidden)()
        )
      }
      
      # 输出头：先做一个 LayerNorm，再线性映射回 dim_x
      self$out_norm <- torch::nn_layer_norm(hidden)
      self$out_proj <- torch::nn_linear(hidden, dim_x)
    },
    
    forward = function(x_t, t_frac) {
      # x_t: [B, dim_x], t_frac: [B,1] 或 [B]
      B <- x_t$size()[1]
      
      # ---- 1. 保证 t_frac 是 [B,1] ----
      if (length(t_frac$shape) == 1) {
        t_frac <- t_frac$unsqueeze(2)  # [B] -> [B,1]
      }
      
      # ---- 2. sinusoidal 时间编码：跟你之前版本保持一致，但 t_dim 可以更大 ----
      t_dim <- self$t_dim
      half  <- as.integer(t_dim / 2)
      
      freqs <- torch::torch_exp(
        torch::torch_arange(
          0, half - 1,
          dtype  = torch::torch_float(),
          device = t_frac$device
        ) * (-log(10000.0) / half)
      )
      # t_frac: [B,1] ; freqs: [1, half] -> args: [B, half]
      args <- t_frac$matmul(freqs$view(c(1, half)))
      t_emb <- torch::torch_cat(
        list(torch::torch_sin(args), torch::torch_cos(args)),
        dim = 2
      )  # [B, t_dim]
      
      # ---- 3. 时间 embedding 通过 MLP -> t_h: [B, hidden] ----
      t_h <- self$time_mlp(t_emb)
      
      # ---- 4. x_t 投影到 hidden 维度 ----
      h <- self$in_proj(x_t)  # [B, hidden]
      
      # ---- 5. 通过多个残差块，每个块都注入 t_h ----
      for (i in seq_len(length(self$blocks))) {
        block <- self$blocks[[i]]
        h <- block(h, t_h)     # [B, hidden]
      }
      
      # ---- 6. 输出头 ----
      h <- self$out_norm(h)
      h <- torch::nnf_silu(h)
      out <- self$out_proj(h)  # [B, dim_x]
      
      out   # 即 eps_pred
    }
  )
}
