# Optional helpers using foreach/doParallel for metric evaluation
parallel_psnr <- function(imgs, refs, cores = parallel::detectCores()/2){
  stopifnot(dim(imgs) == dim(refs))
  cl <- parallel::makeCluster(cores)
  doParallel::registerDoParallel(cl)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  B <- dim(imgs)[1]
  vals <- foreach::foreach(i = 1:B, .combine = c) %dopar% {
    mse <- mean((imgs[i,,,drop=TRUE] - refs[i,,,drop=TRUE])^2)
    10 * log10(1 / mse)
  }
  mean(vals)
}
