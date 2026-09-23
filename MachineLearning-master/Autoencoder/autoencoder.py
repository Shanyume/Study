"""Autoencoder（PyTorch）：MNIST 降维与重构。"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 28*28), nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28))
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def main():
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(latent_dim=32).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        model.train(); total_loss = 0
        for X, _ in train_loader:
            X = X.to(device)
            optimizer.zero_grad()
            recon = model(X)
            loss = criterion(recon, X)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}, recon loss={total_loss/len(train_loader):.6f}")

if __name__ == "__main__":
    main()
