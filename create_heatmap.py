from pathlib import Path

from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms

import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==================================================
# 2. MODEL PATH
# ==================================================

model_path = Path(
    r"C:\\Users\\Anurag Prasad\\Desktop\\High_Resolution_Histopathology_AI\\histology_model.pth"
)

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file was not found:\n{model_path}"
    )


# ==================================================
# 3. CREATE MODEL
# ==================================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ==================================================
# 4. LOAD CHECKPOINT CORRECTLY
# ==================================================

checkpoint = torch.load(
    model_path,
    map_location=device
)

print("Checkpoint loaded.")
print("Checkpoint keys:", checkpoint.keys())


if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ==================================================
# 5. GET CLASS NAMES
# ==================================================

if "classes" in checkpoint:

    class_names = checkpoint["classes"]

else:

    class_names = ["benign", "malignant"]

print("Class names:", class_names)


if "malignant" in class_names:

    malignant_index = class_names.index(
        "malignant"
    )

else:

    malignant_index = 1


# ==================================================
# 6. IMAGE TRANSFORMATION
# ==================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================================================
# 7. INPUT IMAGE FOLDER
# ==================================================

image_folder = Path(
    r"C:\\Users\\Anurag Prasad\\Downloads\\Dataset_image\\BreaKHis 400X\\test\\malignant"
)

if not image_folder.exists():
    raise FileNotFoundError(
        f"Image folder was not found:\n{image_folder}"
    )


# ==================================================
# 8. FIND IMAGES
# ==================================================

image_files = []

for extension in [
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.bmp"
]:
    image_files.extend(
        image_folder.glob(extension)
    )


if len(image_files) == 0:
    raise FileNotFoundError(
        f"No images found in:\n{image_folder}"
    )


# Select the first image
image_path = image_files[0]

print("Selected image:", image_path)


# ==================================================
# 9. OPEN IMAGE
# ==================================================

image = Image.open(image_path).convert("RGB")

width, height = image.size

print("Original image size:", width, "x", height)


# ==================================================
# 10. PATCH SETTINGS
# ==================================================

patch_size = 224
stride = 224


if width < patch_size or height < patch_size:
    raise RuntimeError(
        "The selected image is smaller than "
        "224 x 224 pixels. No patch can be created."
    )


# ==================================================
# 11. HEATMAP SIZE
# ==================================================

number_of_rows = (
    (height - patch_size) // stride
) + 1

number_of_columns = (
    (width - patch_size) // stride
) + 1


heatmap = np.zeros(
    (
        number_of_rows,
        number_of_columns
    ),
    dtype=np.float32
)


# ==================================================
# 12. PREDICT EVERY PATCH
# ==================================================

with torch.no_grad():

    for row, y in enumerate(
        range(
            0,
            height - patch_size + 1,
            stride
        )
    ):

        for column, x in enumerate(
            range(
                0,
                width - patch_size + 1,
                stride
            )
        ):

            patch = image.crop(
                (
                    x,
                    y,
                    x + patch_size,
                    y + patch_size
                )
            )

            input_tensor = transform(patch)

            input_tensor = input_tensor.unsqueeze(0)

            input_tensor = input_tensor.to(device)

            output = model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            malignant_probability = (
                probabilities[0][malignant_index].item()
            )

            heatmap[row, column] = (
                malignant_probability
            )

            print(
                f"Patch at ({x}, {y}) | "
                f"Malignant probability: "
                f"{malignant_probability:.4f}"
            )


# ==================================================
# 13. CREATE OUTPUT FOLDER
# ==================================================

output_folder = Path(
    r"C:\\Users\\Anurag Prasad\\Desktop\\High_Resolution_Histopathology_AI\\outputs\\heatmaps"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# 14. CREATE HEATMAP
# ==================================================

plt.figure(figsize=(10, 8))

plt.imshow(image)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.5,
    extent=(
        0,
        width,
        height,
        0
    ),
    interpolation="nearest"
)

plt.colorbar(
    label="Malignant probability"
)

plt.title(
    "Patch-wise Malignancy Heatmap"
)

plt.axis("off")

plt.tight_layout()


# ==================================================
# 15. SAVE HEATMAP
# ==================================================

output_file = (
    output_folder / "malignancy_heatmap.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nHeatmap created successfully.")
print("Saved at:")
print(output_file)