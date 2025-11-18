// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Utility: row-wise scale and add noise: Xt = sa * X0 + om * N
// [[Rcpp::export]]
arma::mat q_sample_xt_given_x0(const arma::mat& X0,
                               const arma::vec& sa,     // sqrt(alpha_bar_t)
                               const arma::vec& om) {   // sqrt(1 - alpha_bar_t)
  int n = X0.n_rows, d = X0.n_cols;
  arma::mat Xt(n, d, fill::none);
  arma::mat noise = randn<mat>(n, d);
  if(sa.n_elem == 1 && om.n_elem == 1){
    Xt = sa(0) * X0 + om(0) * noise;
    return Xt;
  }
  for (int i=0;i<n;i++){
    double sai = sa( sa.n_elem==1 ? 0 : i );
    double omi = om( om.n_elem==1 ? 0 : i );
    Xt.row(i) = sai * X0.row(i) + omi * noise.row(i);
  }
  return Xt;
}

// DDPM posterior mean μ_θ (vectorized over batch), given eps_pred
// [[Rcpp::export]]
arma::mat ddpm_posterior_mean(const arma::mat& x_t,
                              const arma::mat& eps_pred,
                              const arma::vec& sqrt_alpha_bar_t,
                              const arma::vec& sqrt_alpha_bar_tm1,
                              const arma::vec& beta_t,
                              const arma::vec& one_minus_alpha_bar_t,
                              const arma::vec& one_minus_alpha_bar_tm1) {
  int n = x_t.n_rows, d = x_t.n_cols;
  arma::mat x0_hat(n, d);
  for(int i=0;i<n;i++){
    double sa_t = sqrt_alpha_bar_t( sqrt_alpha_bar_t.n_elem==1 ? 0 : i );
    double om_t = one_minus_alpha_bar_t( one_minus_alpha_bar_t.n_elem==1 ? 0 : i );
    for(int j=0;j<d;j++){
      x0_hat(i,j) = (x_t(i,j) / sa_t) - std::sqrt(om_t) * eps_pred(i,j);
    }
  }
  arma::mat mu(n, d);
  for(int i=0;i<n;i++){
    double sa_tm1 = sqrt_alpha_bar_tm1( sqrt_alpha_bar_tm1.n_elem==1 ? 0 : i );
    double bt = beta_t( beta_t.n_elem==1 ? 0 : i );
    double om_t = one_minus_alpha_bar_t( one_minus_alpha_bar_t.n_elem==1 ? 0 : i );
    double om_tm1 = one_minus_alpha_bar_tm1( one_minus_alpha_bar_tm1.n_elem==1 ? 0 : i );
    double c1 = (sa_tm1 * bt) / om_t;
    double c2 = std::sqrt(1.0 - bt) * (om_tm1 / om_t);
    for(int j=0;j<d;j++){
      mu(i,j) = c1 * x0_hat(i,j) + c2 * x_t(i,j);
    }
  }
  return mu;
}

// DDPM variance (tilde_beta_t) as row vector for broadcast
// [[Rcpp::export]]
arma::vec ddpm_posterior_var(const arma::vec& beta_t,
                             const arma::vec& one_minus_alpha_bar_t,
                             const arma::vec& one_minus_alpha_bar_tm1) {
  int n = beta_t.n_elem;
  arma::vec var(n);
  for(int i=0;i<n;i++){
    double bt = beta_t( n==1 ? 0 : i );
    double om_t = one_minus_alpha_bar_t( one_minus_alpha_bar_t.n_elem==1 ? 0 : i );
    double om_tm1 = one_minus_alpha_bar_tm1( one_minus_alpha_bar_tm1.n_elem==1 ? 0 : i );
    var(i) = (om_tm1 / om_t) * bt;
  }
  return var;
}

// One reverse SDE (VP) Euler-Maruyama step with a provided score(x_t, t)
// Reverse SDE: dx = [-(beta/2) x - beta * score] dt + sqrt(beta) dW, dt > 0 is a small positive step toward t-Δ
// Here we implement a step from t -> t - dt (conceptually), with dt as a small positive scalar.
// [[Rcpp::export]]
arma::mat vp_reverse_sde_step(const arma::mat& x_t,
                              const arma::mat& score_t, // same shape as x_t
                              double beta_t,
                              double dt) {
  int n = x_t.n_rows, d = x_t.n_cols;
  arma::mat noise = randn<mat>(n, d);
  double drift_coeff_x = -0.5 * beta_t * dt;
  double drift_coeff_s = -beta_t * dt;
  double diffusion = std::sqrt(beta_t * dt);
  arma::mat x = x_t + drift_coeff_x * x_t + drift_coeff_s * score_t + diffusion * noise; // drift_coeff_s % score_t 
  return x;
}
