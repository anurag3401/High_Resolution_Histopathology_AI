
from PIL import Image, ImageDraw

import torch
import torch.nn as nn

from torchvision import transforms, models

from pathlib import Path


# ----------------------------------------
# 1. SETTINGS
# ----------------------------------------

IMAGE_PATH = "large_histology_image.png"

MODEL_PATH = "outputs/histology_model.pth"

OUTPUT_PATH = "outputs/prediction_map.png"

PATCH_SIZE = 128

STRIDE = 128

IMAGE_SIZE = 224


# ----------------------------------------
# 2. DEVICE
# ----------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------------------
# 3. LOAD MODEL CHECKPOINT
# ----------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

class_names = checkpoint["class_names"]


# ----------------------------------------
# 4. LOAD MODEL
# ----------------------------------------

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(class_names)
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ----------------------------------------
# 5. IMAGE TRANSFORM
# ----------------------------------------

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ----------------------------------------
# 6. LOAD LARGE IMAGE
# ----------------------------------------

image = Image.open(
    IMAGE_PATH
).convert("RGB")


width, height = image.size


print("Large image size:", width, height)


# ----------------------------------------
# 7. CREATE PREDICTION MAP
# ----------------------------------------

prediction_map = Image.new(
    "RGB",
    (width, height),
    "white"
)

draw = ImageDraw.Draw(
    prediction_map
)


# ----------------------------------------
# 8. PREDICT PATCHES
# ----------------------------------------

patch_count = 0


with torch.no_grad():

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        STRIDE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        ):

            patch = image.crop(
                (
                    x,
                    y,
                    x + PATCH_SIZE,
                    y + PATCH_SIZE
                )
            )


            input_tensor = transform(
                patch
            ).unsqueeze(0).to(device)


            outputs = model(
                input_tensor
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()


            # --------------------------------
            # ASSIGN COLORS
            # --------------------------------

            if predicted_class == 0:

                color = (100, 180, 100)

            else:

                color = (220, 100, 100)


            draw.rectangle(
                [
                    x,
                    y,
                    x + PATCH_SIZE - 1,
                    y + PATCH_SIZE - 1
                ],
                fill=color
            )


            patch_count += 1


print("Total patches predicted:", patch_count)


# ----------------------------------------
# 9. SAVE MAP
# ----------------------------------------

Path("outputs").mkdir(
    parents=True,
    exist_ok=True
)


prediction_map.save(
    OUTPUT_PATH
)


print("Prediction map saved at:", OUTPUT_PATH)