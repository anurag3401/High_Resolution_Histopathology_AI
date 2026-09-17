from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math

# --------------------------------------------------
# 1. Location of patch images
# --------------------------------------------------

patch_dir = Path("outputs/patches")

# --------------------------------------------------
# 2. Find all patches
# --------------------------------------------------

patch_files = sorted(
    patch_dir.glob("*.png")
)

print("Total patches found:", len(patch_files))

if len(patch_files) == 0:
    raise FileNotFoundError(
        "No patches found. Run prepare_patches.py first."
    )

# --------------------------------------------------
# 3. Select how many patches to display
# --------------------------------------------------

number_to_display = min(16, len(patch_files))

selected_patches = patch_files[:number_to_display]

# Calculate grid size
columns = 4
rows = math.ceil(number_to_display / columns)

# --------------------------------------------------
# 4. Display patches
# --------------------------------------------------

plt.figure(figsize=(10, 10))

for index, patch_file in enumerate(selected_patches):

    image = Image.open(patch_file)

    plt.subplot(rows, columns, index + 1)

    plt.imshow(image)

    plt.title(patch_file.stem)

    plt.axis("off")

plt.tight_layout()

# --------------------------------------------------
# 5. Save visualization
# --------------------------------------------------

output_file = Path("outputs/patch_visualization.png")

plt.savefig(output_file, dpi=300)

plt.show()

print("Visualization saved at:", output_file)