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


build_cnn_eps <- function(ch = 32){
  torch::nn_module(
    classname = "cnn_eps",
    initialize = function(){
      # 时间 embedding: 1 -> ch -> 1
      self$emb <- torch::nn_sequential(
        torch::nn_linear(1, ch), torch::nn_silu(),
        torch::nn_linear(ch, 1L)
      )
      self$conv1 <- torch::nn_conv2d(2, ch, kernel_size = 3, padding = 1)
      self$conv2 <- torch::nn_conv2d(ch, ch, kernel_size = 3, padding = 1)
      self$conv3 <- torch::nn_conv2d(ch, 1, kernel_size = 3, padding = 1)
    },
    forward = function(x, t_frac){
      B <- x$size()[1]; H <- x$size()[3]; W <- x$size()[4]
      # t_frac 形状: [B] -> 需要转为 [B, 1] 然后通过 embedding
      if (length(t_frac$shape) == 1) {
        t_frac <- t_frac$unsqueeze(2)  # [B] -> [B, 1]
      }
      em <- self$emb(t_frac)  # [B, 1] -> [B, 1]
      em <- em$view(c(B, 1, 1, 1))$expand(c(B, 1, H, W))  # [B, 1, H, W]
      x_in <- torch::torch_cat(list(x, em), dim = 2)  # [B, 1+1, H, W] = [B, 2, H, W]
      h <- torch::nnf_relu(self$conv1(x_in))
      h <- torch::nnf_relu(self$conv2(h))
      out <- self$conv3(h)
      out
    }
  )
}
