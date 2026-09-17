from pathlib import Path
from PIL import Image

dataset_path = Path(
    r"C:\\Users\\Anurag Prasad\\Downloads\\Dataset_image\\BreaKHis 400X"
)

image_extensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
]

largest_image = None
largest_area = 0

for file in dataset_path.rglob("*"):

    if file.is_file() and file.suffix.lower() in image_extensions:

        try:
            image = Image.open(file)

            width, height = image.size
            area = width * height

            if area > largest_area:
                largest_area = area
                largest_image = file

        except Exception:
            pass

print("Largest image:", largest_image)
print("Image area:", largest_area)

if largest_image is not None:

    image = Image.open(largest_image)

    print("Image size:", image.size)