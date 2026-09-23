"""GRU（PyTorch）时序分类：正弦 vs 余弦 二分类。"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def make_data(n=1000, length=50):
    t = np.linspace(0, 2*np.pi, length)
    X_sin = np.array([np.sin(t + np.random.rand()*0.5) for _ in range(n//2)])
    X_cos = np.array([np.cos(t + np.random.rand()*0.5) for _ in range(n//2)])
    X = np.vstack([X_sin, X_cos])[:, :, None].astype(np.float32)
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

def main():
    X, y = make_data(1000, 50)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model = SimpleGRU()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
    for epoch in range(30):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(Xtr_t), ytr_t)
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(dim=1).numpy()
    print("GRU test acc:", accuracy_score(yte, pred))

if __name__ == "__main__":
    main()
