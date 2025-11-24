// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::depends(torch)]]
#include <Rcpp.h>
// #include <torch/torch.h>
#include <torch.h>
#include <vector>
#include <random>
#include <cmath>

using namespace Rcpp;


torch::Tensor get_timestep_embedding_batch(torch::Tensor t, int dim, torch::Device dev) {
  int half_dim = dim / 2;
  torch::Tensor freqs = torch::exp(
    torch::arange(half_dim, torch::TensorOptions().device(dev).dtype(torch::kFloat32))
    * (-std::log(10000.0f) / half_dim)
  );
  
  torch::Tensor args = t.to(torch::kFloat32).unsqueeze(1) * freqs.unsqueeze(0);
  return torch::cat({torch::sin(args), torch::cos(args)}, 1);
}


struct DoubleConvTimeImpl : torch::nn::Module {
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::Linear t_proj{nullptr};
  
  DoubleConvTimeImpl(int in_ch=1, int out_ch=64, int t_dim=128) {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)));
    t_proj = register_module("t_proj", torch::nn::Linear(t_dim, out_ch));
  }
  
  torch::Tensor forward(torch::Tensor x, torch::Tensor t_emb) {
    auto B = x.size(0);
    auto te = t_proj->forward(t_emb).view({B, -1, 1, 1});
    auto y = conv1->forward(x) + te;
    y = torch::relu(y);
    y = conv2->forward(y);
    y = torch::relu(y);
    return y;
  }
};
TORCH_MODULE(DoubleConvTime);

struct UNetFullImpl : torch::nn::Module {
  DoubleConvTime down1{nullptr}, down2{nullptr};
  DoubleConvTime up1{nullptr}, up2{nullptr};
  torch::nn::Conv2d final{nullptr};
  int t_dim;
  
  UNetFullImpl(int in_ch=1, int out_ch=1, int t_dim_ = 128) : t_dim(t_dim_) {
    down1 = register_module("down1", DoubleConvTime(in_ch, 64, t_dim));
    down2 = register_module("down2", DoubleConvTime(64, 128, t_dim));
    up1   = register_module("up1", DoubleConvTime(128 + 64, 64, t_dim));
    up2   = register_module("up2", DoubleConvTime(64 + in_ch, 64, t_dim));
    final = register_module("final", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, out_ch, 1)));
  }
  
  torch::Tensor forward(torch::Tensor x, torch::Tensor t_vec) {
    auto dev = x.device();
    torch::Tensor t_emb = get_timestep_embedding_batch(t_vec, t_dim, dev); 
    
    auto x1 = down1->forward(x, t_emb);
    auto x2 = down2->forward(torch::max_pool2d(x1, 2), t_emb);
    
    std::vector<int64_t> up_size1 = {x1.size(2), x1.size(3)};
    auto x2_up = torch::nn::functional::interpolate(
      x2,
      torch::nn::functional::InterpolateFuncOptions().size(up_size1).mode(torch::kNearest)
    );
    
    auto x3 = up1->forward(torch::cat({x2_up, x1}, 1), t_emb);
    
    std::vector<int64_t> up_size2 = {x.size(2), x.size(3)};
    auto x3_up = torch::nn::functional::interpolate(
      x3,
      torch::nn::functional::InterpolateFuncOptions().size(up_size2).mode(torch::kNearest)
    );
    
    auto x4 = up2->forward(torch::cat({x3_up, x}, 1), t_emb);
    return final->forward(x4);
  }
};
TORCH_MODULE(UNetFull);


torch::Tensor cosine_noise_schedule(int T, torch::Device dev=torch::kCPU){
  torch::Tensor beta = torch::linspace(1e-4, 0.02, T, torch::kFloat32).to(dev);
  torch::Tensor alpha = 1.0 - beta;
  return alpha;
}

// [[Rcpp::export]]
SEXP create_unet(int in_ch = 1, int out_ch = 1, int t_dim = 128){
  UNetFullImpl* p = new UNetFullImpl(in_ch, out_ch, t_dim);
  return Rcpp::XPtr<UNetFullImpl>(p, true);
}
// [[Rcpp::export]]
SEXP create_optimizer(SEXP net_ptr_sexp, double lr){
  Rcpp::XPtr<UNetFullImpl> net(net_ptr_sexp);
  
  auto opt = new torch::optim::Adam(net->parameters(),
                                    torch::optim::AdamOptions(lr));
  return Rcpp::XPtr<torch::optim::Adam>(opt, true);
}

// [[Rcpp::export]] 
double train_unet(Rcpp::NumericVector x0_r,int T,SEXP net_ptr_sexp,SEXP opt_ptr_sexp)
  {
  Rcpp::XPtr<UNetFullImpl> net(net_ptr_sexp);
  Rcpp::XPtr<torch::optim::Adam> optimizer(opt_ptr_sexp);
  
  Rcpp::IntegerVector dims_r = x0_r.attr("dim");
  std::vector<int64_t> dims(dims_r.begin(), dims_r.end());
  
  torch::Tensor x0 = torch::from_blob((double*)x0_r.begin(),
                                      dims,
                                      torch::kFloat64)
    .to(torch::kFloat32)
    .clone();
    int64_t B = x0.size(0);
                                    
    torch::Device dev = x0.device();
    torch::Tensor alphas = cosine_noise_schedule(T, dev);
    torch::Tensor alphas_cum = torch::cumprod(alphas, 0);
                                      
    std::vector<int64_t> t_vec(B);
    for(int i=0;i<B;i++){
       t_vec[i] = rand() % T;
  }
   torch::Tensor t_tensor = torch::tensor(t_vec, torch::kInt64).to(dev);                                 
  torch::Tensor eps = torch::randn_like(x0);                         
  torch::Tensor alpha_bar = alphas_cum.index_select(0, t_tensor).view({B, 1, 1, 1});
                                      
  torch::Tensor x_t = torch::sqrt(alpha_bar) * x0 + torch::sqrt(1.0 - alpha_bar) * eps;
                                      
  torch::Tensor pred = net->forward(x_t, t_tensor);
                                      
  torch::Tensor loss = torch::mse_loss(pred, eps);
  optimizer->zero_grad();
  loss.backward();
  torch::nn::utils::clip_grad_norm_(net->parameters(), 0.5);
  optimizer->step();
 return loss.item<double>();
}
                                      
// [[Rcpp::export]]
Rcpp::NumericVector sample_diffusion(SEXP net_ptr_sexp, Rcpp::IntegerVector shape, int T = 1000){
  Rcpp::XPtr<UNetFullImpl> net(net_ptr_sexp);
std::vector<int64_t> shape_vec(shape.begin(), shape.end());
torch::Tensor x = torch::randn(shape_vec).to(torch::kFloat32);

torch::Device dev = x.device();
torch::Tensor alphas = cosine_noise_schedule(T, dev);
torch::Tensor betas = 1.0f - alphas;
torch::Tensor alphas_cum = torch::cumprod(alphas,0);

for(int t=T-1;t>=0;--t){
  torch::Tensor t_tensor = torch::full({x.size(0)}, t, torch::kInt64).to(dev);
  torch::Tensor eps_pred = net->forward(x, t_tensor);

  float alpha_t = alphas[t].item<float>();
  float beta_t  = betas[t].item<float>();
  float alpha_bar_t = alphas_cum[t].item<float>();
  float safe_denom = std::max(1e-5f,1.0f-alpha_bar_t);
  
  x = (1.0f/std::sqrt(std::max(1e-5f,alpha_t)))*(x - (beta_t/std::sqrt(safe_denom))*eps_pred);
  if(t>1) x += std::sqrt(beta_t) * torch::randn_like(x);
  
  if(torch::isnan(x).any().item<bool>() || torch::isinf(x).any().item<bool>()){
    Rcpp::Rcout << "NaN/Inf detected at step " << t << ", zeroing x\n";
    x.zero_();
  }
}

x = ((x.clamp(-1.0,1.0)+1.0)/2.0).contiguous();

torch::Tensor x_double = x.to(torch::kFloat64).to(torch::kCPU);
int64_t n = x_double.numel();
Rcpp::NumericVector out(n);
double* ptr = x_double.data_ptr<double>();
for(int64_t i=0;i<n;i++) out[i] = ptr[i];

out.attr("dim") = Rcpp::wrap(std::vector<int64_t>(x.sizes().begin(),x.sizes().end()));
return out;
}
