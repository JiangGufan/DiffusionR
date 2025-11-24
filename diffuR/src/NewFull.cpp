// [[Rcpp::depends(Rcpp)]]
// 我们这里使用“真正的” libtorch，而不是 R 包 torch 的头文件
// 需要在 Makevars 中自己配置 libtorch 的 include / lib 路径
// [[Rcpp::plugins(cpp17)]]

#include <Rcpp.h>
#include <torch/torch.h>

#include <vector>
#include <random>
#include <cmath>

using namespace Rcpp;

// ------------------------- 时间步嵌入 -------------------------

torch::Tensor get_timestep_embedding_batch(const torch::Tensor& t,
                                           int dim,
                                           torch::Device dev) {
  int half_dim = dim / 2;
  // positions: [0, 1, ..., half_dim-1]
  auto positions = torch::arange(
    0,
    half_dim,
    torch::TensorOptions().dtype(torch::kFloat32).device(dev)
  );
  // freqs = exp(-log(10000)/half_dim * positions)
  auto freqs = torch::exp(-std::log(10000.0f) / half_dim * positions);
  
  // t: [B] -> cast to float and add dim 1
  auto t_float = t.to(torch::kFloat32).to(dev);
  auto args = t_float.unsqueeze(1) * freqs.unsqueeze(0);  // [B, half_dim]
  
  // concat sin/cos along last dim -> [B, dim]
  return torch::cat({torch::sin(args), torch::cos(args)}, 1);
}

// ------------------------- DoubleConv + time -------------------------

struct DoubleConvTimeImpl : torch::nn::Module {
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::Linear t_proj{nullptr};
  
  DoubleConvTimeImpl(int in_ch = 1, int out_ch = 64, int t_dim = 128) {
    conv1 = register_module(
      "conv1",
      torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_ch, out_ch, /*kernel_size=*/3).padding(1)
      )
    );
    conv2 = register_module(
      "conv2",
      torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_ch, out_ch, /*kernel_size=*/3).padding(1)
      )
    );
    t_proj = register_module("t_proj", torch::nn::Linear(t_dim, out_ch));
  }
  
  torch::Tensor forward(const torch::Tensor& x,
                        const torch::Tensor& t_emb) {
    auto B  = x.size(0);
    auto te = t_proj->forward(t_emb).view({B, -1, 1, 1});
    auto y  = conv1->forward(x) + te;
    y = torch::relu(y);
    y = conv2->forward(y);
    y = torch::relu(y);
    return y;
  }
};
TORCH_MODULE(DoubleConvTime);

// ------------------------- UNet 结构 -------------------------

struct UNetFullImpl : torch::nn::Module {
  DoubleConvTime down1{nullptr}, down2{nullptr};
  DoubleConvTime up1{nullptr}, up2{nullptr};
  torch::nn::Conv2d final{nullptr};
  int t_dim;
  
  UNetFullImpl(int in_ch = 1, int out_ch = 1, int t_dim_ = 128)
    : t_dim(t_dim_) {
    down1 = register_module("down1", DoubleConvTime(in_ch, 64, t_dim));
    down2 = register_module("down2", DoubleConvTime(64, 128, t_dim));
    up1   = register_module("up1",   DoubleConvTime(128 + 64, 64, t_dim));
    up2   = register_module("up2",   DoubleConvTime(64 + in_ch, 64, t_dim));
    final = register_module(
      "final",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, out_ch, 1))
    );
  }
  
  torch::Tensor forward(const torch::Tensor& x,
                        const torch::Tensor& t_vec) {
    auto dev   = x.device();
    auto t_emb = get_timestep_embedding_batch(t_vec, t_dim, dev);
    
    // 下采样
    auto x1 = down1->forward(x, t_emb);                         // [B,  64, H,   W]
    auto x2 = down2->forward(torch::max_pool2d(x1, 2), t_emb);  // [B, 128, H/2, W/2]
    
    // 上采样 1
    std::vector<int64_t> up_size1 = {x1.size(2), x1.size(3)};
    auto x2_up = torch::nn::functional::interpolate(
      x2,
      torch::nn::functional::InterpolateFuncOptions()
      .size(up_size1)
      .mode(torch::kNearest)
    );                                                          // [B, 128, H, W]
    
    auto x3 = up1->forward(torch::cat({x2_up, x1}, 1), t_emb);   // [B, 64, H, W]
    
    // 上采样 2 -> 恢复到原图大小
    std::vector<int64_t> up_size2 = {x.size(2), x.size(3)};
    auto x3_up = torch::nn::functional::interpolate(
      x3,
      torch::nn::functional::InterpolateFuncOptions()
      .size(up_size2)
      .mode(torch::kNearest)
    );                                                          // [B, 64, H0, W0]
    
    auto x4 = up2->forward(torch::cat({x3_up, x}, 1), t_emb);    // [B, 64, H0, W0]
    
    return final->forward(x4);                                  // [B, out_ch, H0, W0]
  }
};
TORCH_MODULE(UNetFull);

// ------------------------- 噪声 schedule -------------------------

torch::Tensor cosine_noise_schedule(int T,
                                    torch::Device dev = torch::kCPU) {
  auto options = torch::TensorOptions()
  .dtype(torch::kFloat32)
  .device(dev);
  auto beta = torch::linspace(1e-4, 0.02, T, options);
  auto alpha = 1.0 - beta;
  return alpha;
}

// ------------------------- R 接口：创建网络与优化器 -------------------------

// [[Rcpp::export]]
SEXP create_unet(int in_ch = 1,
                 int out_ch = 1,
                 int t_dim = 128) {
  UNetFullImpl* p = new UNetFullImpl(in_ch, out_ch, t_dim);
  // 由 Rcpp::XPtr 托管对象生命周期
  return Rcpp::XPtr<UNetFullImpl>(p, true);
}

// [[Rcpp::export]]
SEXP create_optimizer(SEXP net_ptr_sexp,
                      double lr) {
  Rcpp::XPtr<UNetFullImpl> net(net_ptr_sexp);
  
  auto opt = new torch::optim::Adam(
    net->parameters(),
    torch::optim::AdamOptions(lr)
  );
  return Rcpp::XPtr<torch::optim::Adam>(opt, true);
}

// ------------------------- R 接口：训练一步 -------------------------

// x0_r 维度: dim = c(B, C, H, W)
// [[Rcpp::export]]
double train_unet(Rcpp::NumericVector x0_r,
                  int T,
                  SEXP net_ptr_sexp,
                  SEXP opt_ptr_sexp) {
  
  Rcpp::XPtr<UNetFullImpl> net(net_ptr_sexp);
  Rcpp::XPtr<torch::optim::Adam> optimizer(opt_ptr_sexp);
  
  // 从 R 的 dim 属性读取维度
  Rcpp::IntegerVector dims_r = x0_r.attr("dim");
  std::vector<int64_t> dims(dims_r.begin(), dims_r.end());
  
  // R 的 NumericVector 是 double；转换成 float32 tensor
  auto x0 = torch::from_blob(
    (double*)x0_r.begin(),
    dims,
    torch::TensorOptions().dtype(torch::kFloat64)
  ).to(torch::kFloat32).clone();      // clone 避免与 R 共享内存
  
  auto dev = x0.device();
  int64_t B = x0.size(0);
  
  auto alphas     = cosine_noise_schedule(T, dev);         // [T]
  auto alphas_cum = torch::cumprod(alphas, 0);             // [T]
  
  // 随机采样 t ∈ {0, …, T-1}，每个样本一个
  std::vector<int64_t> t_vec(B);
  for (int i = 0; i < B; ++i) {
    t_vec[i] = std::rand() % T;
  }
  
  auto t_tensor = torch::from_blob(
    t_vec.data(),
    {B},
    torch::TensorOptions().dtype(torch::kInt64)
  ).clone().to(dev);                                    // [B]
  
  auto eps = torch::randn_like(x0);                       // 噪声 ε
  
  auto alpha_bar = alphas_cum.index_select(0, t_tensor)
                             .view({B, 1, 1, 1});                 // [B,1,1,1]
  
  auto x_t = torch::sqrt(alpha_bar) * x0 +
  torch::sqrt(1.0 - alpha_bar) * eps;         // q(x_t | x_0)
  
  auto pred = net->forward(x_t, t_tensor);                // 预测 ε
  
  auto loss = torch::mse_loss(pred, eps);                 // MSE(ε̂, ε)
  
  optimizer->zero_grad();
  loss.backward();
  torch::nn::utils::clip_grad_norm_(net->parameters(), 0.5);
  optimizer->step();
  
  return loss.item<double>();
}

// ------------------------- R 接口：DDPM 采样 -------------------------

// shape 一般是 c(B, C, H, W)
// [[Rcpp::export]]
Rcpp::NumericVector sample_diffusion(SEXP net_ptr_sexp,
                                     Rcpp::IntegerVector shape,
                                     int T = 1000) {
  Rcpp::XPtr<UNetFullImpl> net(net_ptr_sexp);
  
  std::vector<int64_t> shape_vec(shape.begin(), shape.end());
  auto x = torch::randn(
    shape_vec,
    torch::TensorOptions().dtype(torch::kFloat32)
  );
  
  auto dev = x.device();
  auto alphas     = cosine_noise_schedule(T, dev);
  auto betas      = 1.0f - alphas;
  auto alphas_cum = torch::cumprod(alphas, 0);
  
  for (int t = T - 1; t >= 0; --t) {
    auto t_tensor = torch::full(
    {x.size(0)},
    t,
    torch::TensorOptions().dtype(torch::kInt64).device(dev)
    );
    auto eps_pred = net->forward(x, t_tensor);
    
    float alpha_t     = alphas[t].item<float>();
    float beta_t      = betas[t].item<float>();
    float alpha_bar_t = alphas_cum[t].item<float>();
    
    float safe_denom  = std::max(1e-5f, 1.0f - alpha_bar_t);
    float inv_sqrt_at = 1.0f / std::sqrt(std::max(1e-5f, alpha_t));
    
    x = inv_sqrt_at * (x - (beta_t / std::sqrt(safe_denom)) * eps_pred);
    
    if (t > 1) {
      x += std::sqrt(beta_t) * torch::randn_like(x);
    }
    
    if (torch::isnan(x).any().item<bool>() ||
        torch::isinf(x).any().item<bool>()) {
      Rcpp::Rcout << "NaN/Inf detected at step " << t << ", zeroing x\n";
      x.zero_();
    }
  }
  
  // 映射到 [0,1]
  x = ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).contiguous();
  
  auto x_double = x.to(torch::kFloat64).to(torch::kCPU);
  int64_t n = x_double.numel();
  
  Rcpp::NumericVector out(n);
  auto *ptr = x_double.data_ptr<double>();
  for (int64_t i = 0; i < n; ++i) {
    out[i] = ptr[i];
  }
  
  // 把 dim 写回 R
  out.attr("dim") = Rcpp::wrap(
    std::vector<int64_t>(x.sizes().begin(), x.sizes().end())
  );
  return out;
}
