#' @export
psnr <- function(img, ref, max_val = 1){
  # img, ref: arrays [B,H,W] in [0,1]
  mse <- mean((img - ref)^2)
  10 * log10(max_val^2 / mse)
}

#' @export
ssim <- function(img, ref, C1 = 0.01^2, C2 = 0.03^2){
  # simplified SSIM over whole image (not windowed), for demonstration
  mu_x <- mean(img); mu_y <- mean(ref)
  sigma_x2 <- mean((img - mu_x)^2)
  sigma_y2 <- mean((ref - mu_y)^2)
  sigma_xy <- mean((img - mu_x)*(ref - mu_y))
  ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)*(sigma_x2 + sigma_y2 + C2))
}
