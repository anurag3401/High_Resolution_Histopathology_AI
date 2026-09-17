from pathlib import Path

from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms


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
# 11. PREDICT EVERY PATCH
# ==================================================

patch_results = []

patch_number = 0

with torch.no_grad():

    for y in range(
        0,
        height - patch_size + 1,
        stride
    ):

        for x in range(
            0,
            width - patch_size + 1,
            stride
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

            predicted_index = torch.argmax(
                probabilities,
                dim=1
            ).item()

            predicted_label = class_names[
                predicted_index
            ]

            # Find the malignant class index
            if "malignant" in class_names:
                malignant_index = class_names.index(
                    "malignant"
                )
            else:
                malignant_index = 1

            malignant_probability = (
                probabilities[0][malignant_index].item()
            )

            patch_result = {
                "patch_number": patch_number,
                "x": x,
                "y": y,
                "prediction": predicted_label,
                "malignant_probability":
                    malignant_probability
            }

            patch_results.append(
                patch_result
            )

            print(
                f"Patch {patch_number} | "
                f"Position: ({x}, {y}) | "
                f"Prediction: {predicted_label} | "
                f"Malignant probability: "
                f"{malignant_probability:.4f}"
            )

            patch_number += 1


# ==================================================
# 12. FINAL IMAGE-LEVEL RESULT
# ==================================================

if len(patch_results) == 0:
    raise RuntimeError(
        "No patches were created."
    )


average_probability = sum(
    result["malignant_probability"]
    for result in patch_results
) / len(patch_results)


maximum_probability = max(
    result["malignant_probability"]
    for result in patch_results
)


print("\n====================================")
print("FINAL IMAGE-LEVEL RESULT")
print("====================================")

print(
    "Total patches:",
    len(patch_results)
)

print(
    "Average malignant probability:",
    f"{average_probability:.4f}"
)

print(
    "Highest malignant probability:",
    f"{maximum_probability:.4f}"
)


if average_probability >= 0.5:

    print("Final prediction: MALIGNANT")

else:

    print("Final prediction: BENIGN")


print("\nPatch prediction completed successfully.")