import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

# --------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------

train_dir = Path(
    r"C:\\Users\\Anurag Prasad\\Downloads\\Dataset_image\\BreaKHis 400X\\train"
)

test_dir = Path(
    r"C:\\Users\\Anurag Prasad\\Downloads\\Dataset_image\\BreaKHis 400X\\test"
)

# Check whether folders exist
print("Training folder exists:", train_dir.exists())
print("Testing folder exists:", test_dir.exists())

# --------------------------------------------------
# 2. Image transformation
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    root=str(train_dir),
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=str(test_dir),
    transform=transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

# --------------------------------------------------
# 4. Create data loaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# 5. Select CPU or GPU
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# --------------------------------------------------
# 6. Load pretrained ResNet18
# --------------------------------------------------

model = models.resnet18(weights="DEFAULT")

# Replace the final layer for two classes
number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    2
)

model = model.to(device)

# --------------------------------------------------
# 7. Loss function and optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# --------------------------------------------------
# 8. Training
# --------------------------------------------------

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

# --------------------------------------------------
# 9. Save trained model
# --------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes
    },
    "histology_model.pth"
)

print("Model saved as histology_model.pth")