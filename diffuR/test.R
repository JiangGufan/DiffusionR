library(torch)
library(coro)
library(Rcpp)
library(dslabs)

Rcpp::sourceCpp("~/MY_RUC/Rcoding/Rcpp/finalProj/diffuR/cnn.cpp")
mnist <- dslabs::read_mnist()

imgs <- mnist$train$images   
labels <- mnist$train$labels
N <- nrow(imgs)

x_array <- array(0, dim = c(N, 1, 28, 28))
for (i in 1:N) {
  mat <- matrix(imgs[i, ], nrow = 28, byrow = TRUE)  
  mat <- mat / 255
  mat <- mat * 2 - 1
  x_array[i, 1, , ] <- mat
}
x_train_tensor <- torch_tensor(x_array, dtype = torch_float())
train_ds <- tensor_dataset(x_train_tensor)
batch_size <- 32
train_loader <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)

net_ptr  <- create_unet(in_ch = 1, out_ch = 1, t_dim = 128)
optim_ptr <- create_optimizer(net_ptr, lr = 1e-4)

epochs  <- 5
T_steps <- 200

for (epoch in 1:epochs) {
  total_loss  <- 0
  batch_count <- 0
  t0 <- Sys.time()
  
  coro::loop(for (batch in train_loader) {
    x_batch_torch <- batch[[1]]$to(dtype = torch_float())   
    x_batch <- as.array(x_batch_torch)                      
    
    B <- dim(x_batch)[1]
    x_batch_perm <- aperm(x_batch, c(4, 3, 2, 1))          
    x_vec <- as.numeric(x_batch_perm)
    attr(x_vec, "dim") <- c(B, 1, 28, 28)
    loss_val <- train_unet(x_vec, T_steps, net_ptr, optim_ptr)
    
    total_loss  <- total_loss + loss_val
    batch_count <- batch_count + 1
    
    if (batch_count %% 200 == 0) {
      cat(sprintf("epoch %d batch %d avg_loss_so_far %.6f\n",
                  epoch, batch_count, total_loss / batch_count))
    }
  })
  
  t1 <- Sys.time()
  cat(sprintf("Epoch %d finished — avg loss %.6f — epoch time %s\n",
              epoch, total_loss / batch_count, format(t1 - t0)))

}

 gen <- sample_diffusion(net_ptr, c(16,1,28,28), T = 200)

 # img <- as.numeric(gen)            
 # dim(img) <- c(28, 28)            
 # 
 # image(t(apply(img, 2, rev)), col=gray.colors(256), axes=FALSE)

 par(mfrow=c(4,4), mar=c(1,1,1,1))

 for (i in 1:16) {
   img <- matrix(gen[((i-1)*28*28 + 1):(i*28*28)], nrow=28, ncol=28)
   img <- t(apply(img, 2, rev))
   image(img, col=gray(seq(0,1,length=256)), axes=FALSE)
 }

 
 # 计算前 100 个 batch 的平均时间
 # t0 <- Sys.time()
 # cnt <- 0
 # coro::loop(for(batch in train_loader){
 #   if(cnt >= 100) break
 #   x_batch <- batch[[1]]$to(dtype = torch_float())
 #   x_vec <- as.numeric(x_batch); dim(x_vec) <- dim(x_batch)
 #   loss_val <- train_unet(x_vec, T, net_ptr, optim_ptr)
 #   cnt <- cnt + 1
 # })
 # t1 <- Sys.time()
 # 
 # avg_sec_per_batch <- as.numeric(difftime(t1,t0,units="secs")) / cnt
 # est_epoch_secs <- avg_sec_per_batch * (nrow(x_train) / batch_size)
 # cat(sprintf("approx %.3f sec/batch -> est epoch time %.1f sec (%.2f min)\n",
 #             avg_sec_per_batch, est_epoch_secs, est_epoch_secs/60))
 # 