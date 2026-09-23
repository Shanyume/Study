"""CNN（PyTorch）MNIST 分类简洁实现。"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

def main():
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(3):
        model.train(); total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}, loss={total_loss/len(train_loader):.4f}")
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X.to(device)).argmax(dim=1).cpu().numpy()
            preds.extend(pred); trues.extend(y.numpy())
    print("CNN MNIST test acc:", accuracy_score(trues, preds))

if __name__ == "__main__":
    main()
