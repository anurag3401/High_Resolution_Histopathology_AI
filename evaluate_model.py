import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

# --------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------

test_dir = Path(
    r"C:\\Users\\Anurag Prasad\\Downloads\\Dataset_image\\BreaKHis 400X\\test"
)

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
# 3. Load test dataset
# --------------------------------------------------

test_dataset = datasets.ImageFolder(
    root=str(test_dir),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("Classes:", test_dataset.classes)
print("Test images:", len(test_dataset))

# --------------------------------------------------
# 4. Select device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# --------------------------------------------------
# 5. Create the same model architecture
# --------------------------------------------------

model = models.resnet18(weights=None)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    2
)

# --------------------------------------------------
# 6. Load trained model
# --------------------------------------------------

checkpoint = torch.load(
    "histology_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------

correct = 0
total = 0

class_correct = [0, 0]
class_total = [0, 0]

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i].item()

            class_total[label] += 1

            if predicted[i] == labels[i]:
                class_correct[label] += 1

accuracy = 100 * correct / total

print("\nOverall test accuracy:", round(accuracy, 2), "%")

for i, class_name in enumerate(test_dataset.classes):

    class_accuracy = (
        100 * class_correct[i] / class_total[i]
        if class_total[i] > 0 else 0
    )

    print(
        class_name,
        "accuracy:",
        round(class_accuracy, 2),
        "%"
    )