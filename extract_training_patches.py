from pathlib import Path
from PIL import Image

# Change this path if required
dataset_path = Path(
    r"C:\\Users\\Anurag Prasad\\Downloads\\Dataset_image\\BreaKHis 400X"
)

output_path = Path("patch_dataset")

patch_size = 224
stride = 224

classes = ["benign", "malignant"]

for split in ["train", "test"]:
    for class_name in classes:

        input_folder = dataset_path / split / class_name
        output_folder = output_path / split / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        image_files = list(input_folder.glob("*"))

        for image_index, image_file in enumerate(image_files):

            try:
                image = Image.open(image_file).convert("RGB")
                width, height = image.size

                patch_number = 0

                for y in range(0, height - patch_size + 1, stride):
                    for x in range(0, width - patch_size + 1, stride):

                        patch = image.crop(
                            (x, y, x + patch_size, y + patch_size)
                        )

                        output_file = (
                            output_folder
                            / f"{image_file.stem}_patch_{patch_number}.png"
                        )

                        patch.save(output_file)
                        patch_number += 1

                print(
                    f"Processed: {image_file.name} | "
                    f"Patches: {patch_number}"
                )

            except Exception as error:
                print(f"Skipped {image_file}: {error}")

print("Patch extraction completed.")