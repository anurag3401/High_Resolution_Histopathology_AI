import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

model_path = r"C:\\Users\\Anurag Prasad\\Desktop\\High_Resolution_Histopathology_AI\\histology_model.pth"

test_folder = r"C:\\Users\\Anurag Prasad\\Downloads\\Dataset_image\\BreaKHis 400X\\test"

output_csv = r"C:\\Users\\Anurag Prasad\\Desktop\\High_Resolution_Histopathology_AI\\test_results.csv"

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

checkpoint = torch.load(model_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])

class_names = checkpoint["classes"]

model.eval()

print("Model loaded successfully")
print("Classes:", class_names)

# --------------------------------------------------
# Image preprocessing
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
# Test all images
# --------------------------------------------------

true_labels = []
predicted_labels = []
results = []

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

with torch.no_grad():

    for actual_class in ["benign", "malignant"]:

        class_folder = os.path.join(test_folder, actual_class)

        if not os.path.exists(class_folder):
            print("Folder not found:", class_folder)
            continue

        image_files = [
            file for file in os.listdir(class_folder)
            if file.lower().endswith(valid_extensions)
        ]

        print(f"\nTesting {actual_class} images: {len(image_files)}")

        for image_name in image_files:

            image_path = os.path.join(class_folder, image_name)

            try:
                image = Image.open(image_path).convert("RGB")

                image_tensor = transform(image)
                image_tensor = image_tensor.unsqueeze(0)
                image_tensor = image_tensor.to(device)

                output = model(image_tensor)

                probabilities = torch.softmax(output, dim=1)

                predicted_index = torch.argmax(probabilities, dim=1).item()
                predicted_class = class_names[predicted_index]

                confidence = probabilities[0][predicted_index].item()

                true_labels.append(actual_class)
                predicted_labels.append(predicted_class)

                results.append({
                    "image_name": image_name,
                    "image_path": image_path,
                    "actual_label": actual_class,
                    "predicted_label": predicted_class,
                    "confidence": confidence
                })

                print(
                    f"{image_name} | "
                    f"Actual: {actual_class} | "
                    f"Predicted: {predicted_class} | "
                    f"Confidence: {confidence:.4f}"
                )

            except Exception as error:
                print("Error processing:", image_path)
                print(error)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

if len(true_labels) == 0:
    print("No images were tested.")
    raise SystemExit

accuracy = accuracy_score(true_labels, predicted_labels)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=["benign", "malignant"]
)

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print("Total images tested:", len(true_labels))
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print("Rows = Actual labels")
print("Columns = Predicted labels")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=["benign", "malignant"],
        zero_division=0
    )
)

# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

print("\nDetailed results saved to:")
print(output_csv)