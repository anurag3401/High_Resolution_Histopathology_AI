import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# --------------------------------------------------
# 1. Select an image
# --------------------------------------------------

image_path = Path(
    r"C:\\Users\\Anurag Prasad\\Downloads\\Dataset_image\\BreaKHis 400X\\test\\malignant"
)

# Find the first image inside the benign folder
image_files = [
    file for file in image_path.rglob("*")
    if file.suffix.lower() in [
        ".jpg", ".jpeg", ".png", ".tif", ".tiff"
    ]
]

if len(image_files) == 0:
    raise FileNotFoundError("No image found in the selected folder.")

image_file = image_files[0]

print("Image selected:", image_file)

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
# 3. Load image
# --------------------------------------------------

image = Image.open(image_file).convert("RGB")

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

# --------------------------------------------------
# 4. Select device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# 5. Create model
# --------------------------------------------------

model = models.resnet18(weights=None)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    2
)

# --------------------------------------------------
# 6. Load trained weights
# --------------------------------------------------

checkpoint = torch.load(
    "histology_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

classes = checkpoint["classes"]

model = model.to(device)
model.eval()

# --------------------------------------------------
# 7. Predict
# --------------------------------------------------

image_tensor = image_tensor.to(device)

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

confidence = probabilities[0][predicted_class].item() * 100

print("\nPrediction:", classes[predicted_class])
print("Confidence:", round(confidence, 2), "%")