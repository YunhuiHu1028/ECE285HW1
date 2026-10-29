import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from ignite.metrics import FID, InceptionScore

# Dataset and output paths
IMG_SIZE = 256
TRAIN_DIR = r"C:\Users\jia65\Desktop\ECE285\hw1\chest_xray\train"
TEST_DIR = r"C:\Users\jia65\Desktop\ECE285\hw1\chest_xray\test"
OUT_DIR = r"C:\Users\jia65\Desktop\ECE285\hw1\output"

# Convert images to grayscale, resize, crop, and normalize to [0, 1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])


# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        # Downsampling block for the encoder
        def down(input_ch, output_ch):
            return nn.Sequential(
                nn.Conv2d(input_ch, output_ch, 4, 2, 1),
                nn.BatchNorm2d(output_ch),
                nn.LeakyReLU()
            )

        # Upsampling block for the decoder
        def up(input_ch, output_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(input_ch, output_ch, 4, 2, 1),
                nn.BatchNorm2d(output_ch),
                nn.LeakyReLU()
            )

        # Bottleneck feature map: C x S x S
        self.C, self.S = 512, 4
        flat_dim = self.C * self.S * self.S

        # Extract image features: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            down(1, 32),
            down(32, 64),
            down(64, 128),
            down(128, 256),
            down(256, 512),
            down(512, self.C),
        )

        # Predict the mean and log variance of the latent distribution
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Map latent vectors back to feature maps
        self.fc_decoder = nn.Linear(latent_dim, flat_dim)

        # Reconstruct images: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            up(self.C, 512),
            up(512, 256),
            up(256, 128),
            up(128, 64),
            up(64, 32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.latent_dim = latent_dim

    # Encode images into latent distribution parameters
    def encode(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)
        return mu, logvar

    # Sample latent vectors using the reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    # Decode latent vectors into reconstructed images
    def decode(self, z):
        h = self.fc_decoder(z).view(-1, self.C, self.S, self.S)
        return self.decoder(h)

    # Forward pass through the VAE
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Compute reconstruction loss and KL divergence
def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    B = x.size(0)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / B
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return recon + beta * kl, recon, kl


def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img.cpu()).convert("RGB")
        pil_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(pil_img))
    return torch.stack(arr)


def evaluate_fid_is(model, real_loader, device):
    fid_metric = FID(device=device)
    is_metric = InceptionScore(device=device)
    model.eval()
    with torch.no_grad():
        for i, (real, _) in enumerate(real_loader):
            z = torch.randn(real.size(0), model.latent_dim, device=device)
            fake = model.decode(z)
            fake_299 = interpolate(fake).to(device)
            real_299 = interpolate(real).to(device)
            fid_metric.update((fake_299, real_299))
            is_metric.update(fake_299)
            if i % 10 == 0:
                print(f"eval batch {i}/{len(real_loader)}")
    return fid_metric.compute(), is_metric.compute()


if __name__ == "__main__":
    torch.manual_seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_set = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    test_set = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    x, _ = next(iter(train_loader))
    print(x.shape, x.min().item(), x.max().item())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(latent_dim=128).to(device)
    epochs = 50
    learning_rate = 1e-4
    beta = 1
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    fixed_z = torch.randn(64, model.latent_dim, device=device)
    fixed_x, _ = next(iter(test_loader))
    fixed_x = fixed_x[:8].to(device)

    history = {"train_loss": [], "train_recon": [], "train_kl": [], "test_loss": []}

    for i in range(epochs):
        model.train()
        sum_loss = sum_recon = sum_kl = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optim.step()

            sum_loss += loss.item()
            sum_recon += recon.item()
            sum_kl += kl.item()

        n = len(train_loader)
        history["train_loss"].append(sum_loss / n)
        history["train_recon"].append(sum_recon / n)
        history["train_kl"].append(sum_kl / n)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                loss, _, _ = vae_loss(x_hat, x, mu, logvar, beta)
                test_loss += loss.item()
            history["test_loss"].append(test_loss / len(test_loader))

            samples = model.decode(fixed_z)
            save_image(samples, f"{OUT_DIR}/samples_epoch{i+1:03d}.png", nrow=8)

            recon_x, _, _ = model(fixed_x)
            save_image(torch.cat([fixed_x, recon_x]), f"{OUT_DIR}/recon_epoch{i+1:03d}.png", nrow=8)

        print(f"Epoch {i+1:3d}/{epochs} | "
              f"train {history['train_loss'][-1]:.1f} "
              f"(recon {history['train_recon'][-1]:.1f}, kl {history['train_kl'][-1]:.1f}) | "
              f"test {history['test_loss'][-1]:.1f}")

    torch.save(model.state_dict(), f"{OUT_DIR}/vae.pt")

    ep = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["test_loss"], label="test")
    axes[0].set_title("Total loss (-ELBO)")
    axes[0].legend()
    axes[1].plot(ep, history["train_recon"])
    axes[1].set_title("Reconstruction (BCE)")
    axes[2].plot(ep, history["train_kl"])
    axes[2].set_title("KL divergence")
    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/loss_curves.png", dpi=150)
    plt.close()

    model.eval()
    with torch.no_grad():
        samples = model.decode(torch.randn(64, model.latent_dim, device=device))
    save_image(samples, f"{OUT_DIR}/generated_final.png", nrow=8)
    real_examples, _ = next(iter(DataLoader(train_set, batch_size=64, shuffle=True)))
    save_image(real_examples, f"{OUT_DIR}/real_examples.png", nrow=8)

    real_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=0)
    fid, is_score = evaluate_fid_is(model, real_loader, device)
    print(f"FID: {fid:.4f}")
    print(f"IS : {is_score:.4f}")

    with open(f"{OUT_DIR}/metrics.txt", "w") as f:
        f.write(f"FID: {fid:.4f}\nIS: {is_score:.4f}\n")