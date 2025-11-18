#' Linear and cosine beta schedules
#' @export
beta_linear <- function(T = 1000, beta_min = 1e-4, beta_max = 0.02){
  beta <- seq(beta_min, beta_max, length.out = T)
  alpha <- 1 - beta
  alpha_bar <- cumprod(alpha)
  list(beta = beta,
       alpha = alpha,
       alpha_bar = alpha_bar,
       sqrt_alpha_bar = sqrt(alpha_bar),
       sqrt_one_minus_alpha_bar = sqrt(1 - alpha_bar))
}

#' @export
beta_cosine <- function(T = 1000, s = 0.008){
  # from Nichol & Dhariwal 2021
  t <- (0:T)/T
  f <- function(u) { (cos((u + s)/(1+s) * pi/2))^2 }
  alpha_bar <- pmax(f(t)/f(0), 1e-4)
  beta <- pmin(1 - alpha_bar[-1]/alpha_bar[-length(alpha_bar)], 0.999)
  alpha <- 1 - beta
  alpha_bar <- cumprod(alpha)
  list(beta = beta,
       alpha = alpha,
       alpha_bar = alpha_bar,
       sqrt_alpha_bar = sqrt(alpha_bar),
       sqrt_one_minus_alpha_bar = sqrt(1 - alpha_bar))
}
