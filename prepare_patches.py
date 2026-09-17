from PIL import Image
from pathlib import Path

# --------------------------------------------------
# 1. Input image
# --------------------------------------------------

image_path = Path("large_histology_image.png")

# --------------------------------------------------
# 2. Output folder
# --------------------------------------------------

output_dir = Path("outputs/patches")
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3. Patch settings
# --------------------------------------------------

PATCH_SIZE = 128
STRIDE = 128

# --------------------------------------------------
# 4. Load image
# --------------------------------------------------

image = Image.open(image_path).convert("RGB")

width, height = image.size

print("Image size:", image.size)

patch_number = 0

# --------------------------------------------------
# 5. Extract patches
# --------------------------------------------------

for y in range(0, height - PATCH_SIZE + 1, STRIDE):

    for x in range(0, width - PATCH_SIZE + 1, STRIDE):

        patch = image.crop(
            (
                x,
                y,
                x + PATCH_SIZE,
                y + PATCH_SIZE
            )
        )

        patch_path = output_dir / f"patch_{patch_number}.png"

        patch.save(patch_path)

        patch_number += 1

print("Total patches created:", patch_number)