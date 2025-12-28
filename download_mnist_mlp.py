import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#calcolo accuracy digitale
def test_step(validation_data, model, criterion, device="cpu"):
    total_loss, correct, total = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in validation_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = total_loss / len(validation_data.dataset)
    test_acc = 100.0 * correct / total
    return test_loss, 100.0 - test_acc, test_acc

class MLP_Crossbar(nn.Module):
    def __init__(self, size):
        """
        size = dimensione input (es: 64, 121, 256)
        """
        super().__init__()

        if size == 64:           # immagini 8×8
            layers = [nn.Linear(64, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 10)]

        elif size == 121:        # immagini 11×11
            layers = [nn.Linear(121, 121), nn.ReLU(),
                    nn.Linear(121, 64), nn.ReLU(),
                    nn.Linear(64, 10)]

        elif size == 144:        # immagini 12×12
            layers = [nn.Linear(144, 144), nn.ReLU(),
                    nn.Linear(144, 72), nn.ReLU(),
                    nn.Linear(72, 10)]

        elif size == 256:        # immagini 16×16
            layers = [
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 10)
            ]
            
        elif size == 784:        # immagini 28×28
            layers = [
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 10)
            ]


        else:
            raise ValueError(f" Architettura non definita per size={size}")

        self.layers = nn.Sequential(
            nn.Flatten(),
            *layers,
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

# Training generico

def train_model(model, train_loader, test_loader, device, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    print(f"\n Training on: {device}")

    # modello 
    architectures = {
        "64":  (8, 8),   # 8×8 , 64 input
        "121": (11, 11), # 11×11 , 121 input
        "144": (12, 12), # 12×12 , 144 input
        "256": (16, 16),  # 16×16 , 256 input
        "784": (28, 28)  # 28×28 , 784 input
    }

    for name, (h, w) in architectures.items():
        print(f" Training modello MLP_{name} (input {h}x{w})")

        transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(), transforms.Lambda(lambda t: t / (t.abs().max() * 1.2 + 1e-12))
    ])


        train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
        test_set  = datasets.MNIST("data", train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        test_loader  = DataLoader(test_set, batch_size=256, shuffle=False)

        model = MLP_Crossbar(size=h * w).to(device)
        
        acc = train_model(model, train_loader, test_loader, device)
        
        print(f" Accuracy test modello {name}: {acc:.2f}%")
        filename = f"mnist_mlp_{name}.pth"
        torch.save(model.state_dict(), filename)

        print(f" Salvato: {filename}")

    print("\n Training completato per tutte le architetture.\n")

if __name__ == "__main__":
    main()
