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

build_mlp_eps <- function(dim_x, hidden = 256, t_dim = 16){
  torch::nn_module(
    classname = "mlp_eps",
    
    initialize = function(){
      self$t_dim <- t_dim
      
      # 时间嵌入先过一层 MLP
      self$time_mlp <- torch::nn_sequential(
        torch::nn_linear(t_dim, hidden),
        torch::nn_silu()
      )
      
      # 主网络：输入 = x_t (dim_x) + time_hidden (hidden)
      self$net <- torch::nn_sequential(
        torch::nn_linear(dim_x + hidden, hidden), torch::nn_silu(),
        torch::nn_linear(hidden, hidden),        torch::nn_silu(),
        torch::nn_linear(hidden, dim_x)
      )
    },
    
    forward = function(x_t, t_frac){
      # x_t: [B, dim_x], t_frac: [B,1]
      B <- x_t$size()[1]
      t_dim <- self$t_dim
      half <- as.integer(t_dim/2)

      # ===== sinusoidal 时间编码 =====
      # freqs: [half]
      freqs <- torch::torch_exp(
        torch::torch_arange(
          0, half - 1,
          dtype = torch::torch_float(),
          device = t_frac$device
        ) * (-log(10000.0)/half)
      )
      # t_frac: [B,1], freqs: [1,half] -> args: [B,half]
      args <- t_frac$matmul(freqs$view(c(1, half)))
      t_emb <- torch::torch_cat(
        list(torch::torch_sin(args), torch::torch_cos(args)),
        dim = 2
      ) # [B, t_dim]

      # ===== 时间嵌入通过一层 MLP =====
      t_h <- self$time_mlp(t_emb)  # [B, hidden]

      # 拼到 x_t 后面
      h_in <- torch::torch_cat(list(x_t, t_h), dim = 2)  # [B, dim_x + hidden]
      self$net(h_in)  # 输出 [B, dim_x]，即 eps_pred
    }
  )
}
