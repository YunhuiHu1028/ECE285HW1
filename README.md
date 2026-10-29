# ECE285HW1

I trained a convolutional variational autoencoder (VAE) on the Kaggle chest X-ray pneumonia dataset using the standard ELBO ($\beta = 1$), with $256 \times 256$ grayscale images encoded into a 128-dimensional latent space. Images generated from the standard normal prior capture the coarse chest anatomy but remain blurry, achieving an FID of 0.1371 and an Inception Score of 1.5057 with PyTorch-Ignite.
