# 🔬 High-Resolution Histopathology AI

A deep-learning-based web application for classifying breast histopathology images as **benign** or **malignant**, with a patch-based malignancy heatmap for visual analysis.

The project uses a **ResNet18 convolutional neural network** and provides an attractive **Streamlit dashboard** where users can upload a new histopathology image and view the model prediction, class probabilities, confidence score, and patch-level heatmap.

> **Important:** This project is developed for academic and research purposes. It is not a clinically approved medical diagnostic system.

---

## 📌 Project Overview

Breast cancer diagnosis commonly involves examining histopathology images under a microscope. These images contain complex tissue structures and cellular patterns that may be difficult to analyze manually.

This project aims to use deep learning to classify histopathology images into two categories:

* **Benign:** Non-cancerous tissue
* **Malignant:** Cancerous tissue

The trained model analyzes an input image and predicts the class based on the visual features learned during training.

In addition to whole-image classification, the project divides an image into smaller patches and predicts the malignancy probability of each patch. These patch-level predictions are combined to generate a heatmap that shows regions receiving higher or lower malignant scores from the model.

---

## 🎯 Objectives

The main objectives of this project are:

1. To develop a deep-learning model for benign and malignant histopathology image classification.
2. To use a ResNet18 architecture for image feature extraction and classification.
3. To evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
4. To test the model on new images.
5. To generate a patch-based malignancy heatmap.
6. To create an interactive Streamlit dashboard for easy model usage.
7. To provide a simple interface for viewing predictions and patch-level results.

---

## 🧠 Technologies Used

| Technology   | Purpose                                         |
| ------------ | ----------------------------------------------- |
| Python       | Main programming language                       |
| PyTorch      | Deep-learning framework                         |
| Torchvision  | ResNet18 architecture and image transformations |
| ResNet18     | Image classification model                      |
| PIL          | Image loading and processing                    |
| NumPy        | Numerical operations                            |
| Pandas       | Tabular data and CSV results                    |
| Matplotlib   | Heatmap generation                              |
| Scikit-learn | Model evaluation metrics                        |
| Streamlit    | Interactive web dashboard                       |
| Git/GitHub   | Version control and project hosting             |

---

## 🗂️ Project Structure

```text
High_Resolution_Histopathology_AI/
│
├── app.py
├── histology_model.pth
├── test_many_images.py
├── predict_new_image.py
├── create_heatmap.py
├── test_results.csv
├── README.md
│
├── outputs/
│   └── heatmaps/
│       └── malignancy_heatmap.png
│
└── new_images/
```

### File Descriptions

#### `app.py`

Main Streamlit application. It provides:

* Image upload
* Whole-image prediction
* Class probabilities
* Confidence score
* Patch-level predictions
* Patch grid visualization
* Malignancy heatmap
* Patch statistics
* CSV download
* Heatmap download

#### `histology_model.pth`

Saved PyTorch model checkpoint containing:

* Trained model weights
* Class names
* Model state dictionary

#### `test_many_images.py`

Tests multiple images from benign and malignant folders and calculates:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report

#### `predict_new_image.py`

Predicts the class of a single new image that was not part of the training or testing dataset.

#### `create_heatmap.py`

Divides an image into patches and creates a patch-based malignancy heatmap.

#### `outputs/heatmaps/`

Stores generated heatmap images.

---

## 📊 Dataset

The project was tested using the **BreaKHis 400X dataset structure**, containing benign and malignant histopathology images.

The dataset was organized into the following folders:

```text
BreaKHis 400X/
│
├── train/
│   ├── benign/
│   └── malignant/
│
└── test/
    ├── benign/
    └── malignant/
```

The folder names are used as the actual class labels:

```text
benign
malignant
```

The model checkpoint stores the class order as:

```python
["benign", "malignant"]
```

### Dataset Organization

* Images inside `train/benign` represent benign samples.
* Images inside `train/malignant` represent malignant samples.
* Images inside `test/benign` represent benign test samples.
* Images inside `test/malignant` represent malignant test samples.

For a reliable evaluation, test images should not be used during model training.

---

## ⚙️ How the Model Works

The complete workflow is:

```text
Input Histopathology Image
          ↓
Image Preprocessing
          ↓
ResNet18 Model
          ↓
Feature Extraction
          ↓
Benign/Malignant Classification
          ↓
Class Probabilities
          ↓
Prediction and Confidence
          ↓
Patch-Based Heatmap
```

---

## 🔄 Step-by-Step Working

### 1. Image Upload

The user uploads a histopathology image through the Streamlit dashboard.

Supported image formats include:

* PNG
* JPG
* JPEG
* BMP
* TIFF

The uploaded image is converted into RGB format.

---

### 2. Image Preprocessing

Before sending the image to the model, it is resized to:

```text
224 × 224 pixels
```

The image is then converted into a PyTorch tensor.

Normalization is applied using:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

The preprocessing step makes the input compatible with the ResNet18 model.

---

### 3. ResNet18 Classification

The project uses the ResNet18 architecture.

ResNet18 is a convolutional neural network that learns visual features such as:

* Edges
* Textures
* Tissue patterns
* Cellular structures
* Spatial patterns

The final fully connected layer is modified to produce two outputs:

```text
Output 1 → Benign
Output 2 → Malignant
```

The model output is converted into probabilities using the softmax function.

For example:

```text
Benign probability:    0.08
Malignant probability: 0.92
```

The class with the higher probability is selected as the predicted class.

---

## 🧮 Prediction Logic

The model produces two output scores:

```python
output = model(image_tensor)
```

These scores are converted into probabilities:

```python
probabilities = torch.softmax(output, dim=1)
```

The predicted class is selected using:

```python
predicted_index = torch.argmax(probabilities, dim=1).item()
```

The final result contains:

* Predicted class
* Model confidence
* Benign probability
* Malignant probability

---

## 🔥 Patch-Based Heatmap

The heatmap feature divides the original image into smaller patches.

The current patch settings are:

```text
Patch size: 224 × 224 pixels
Stride: 224 pixels
```

Each patch is processed independently by the model.

For every patch, the application calculates:

* Patch location
* Predicted class
* Confidence
* Benign probability
* Malignant probability

The malignant probability of each patch is then placed at its corresponding location in the heatmap.

---

## 🧩 Heatmap Generation Process

```text
Original Image
      ↓
Divide Image into Patches
      ↓
Predict Each Patch
      ↓
Calculate Malignant Probability
      ↓
Place Probability at Patch Location
      ↓
Combine All Patch Scores
      ↓
Generate Heatmap
```

### Heatmap Interpretation

The heatmap generally represents the model's malignant probability response:

* Warmer colors indicate higher malignant probability.
* Cooler colors indicate lower malignant probability.
* Red patch borders indicate malignant predictions.
* Green patch borders indicate benign predictions.

The heatmap is a model-based visualization. It should not be interpreted as a confirmed tumor-location map.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard developed using Streamlit.

The dashboard provides the following sections:

### 1. Image Upload

Users can upload a new histopathology image directly from the browser.

### 2. Image Information

The dashboard displays:

* Image width
* Image height
* Patch size
* Processing device

### 3. Model Prediction

The dashboard shows:

* Predicted class
* Confidence score
* Benign probability
* Malignant probability

### 4. Patch Analysis

The dashboard displays:

* Total number of patches
* Number of benign patches
* Number of malignant patches
* Average malignant probability
* Highest malignant probability

### 5. Visualization

The dashboard provides:

* Original image
* Patch location image
* Combined heatmap
* Individual patch predictions

### 6. Download Options

Users can download:

* Generated heatmap
* Patch prediction CSV file

---

## 📈 Model Evaluation

The model was evaluated on:

```text
Total images tested: 545
```

The test dataset contained:

```text
Benign images: 176
Malignant images: 369
```

### Overall Accuracy

```text
Accuracy: 94.13%
```

This means that the model correctly classified approximately 94 out of every 100 test images.

### Confusion Matrix

```text
[[149  27]
 [  5 364]]
```

Rows represent actual classes, and columns represent predicted classes.

| Actual Class | Predicted Benign | Predicted Malignant |
| ------------ | ---------------: | ------------------: |
| Benign       |              149 |                  27 |
| Malignant    |                5 |                 364 |

### Interpretation

* 149 benign images were correctly classified as benign.
* 27 benign images were incorrectly classified as malignant.
* 5 malignant images were incorrectly classified as benign.
* 364 malignant images were correctly classified as malignant.

### Classification Report

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------: | -----: | -------: | ------: |
| Benign           |      0.97 |   0.85 |     0.90 |     176 |
| Malignant        |      0.93 |   0.99 |     0.96 |     369 |
| Accuracy         |         — |      — |     0.94 |     545 |
| Macro Average    |      0.95 |   0.92 |     0.93 |     545 |
| Weighted Average |      0.94 |   0.94 |     0.94 |     545 |

---

## 📌 Important Evaluation Findings

### Malignant Recall

The malignant recall was approximately:

```text
0.99
```

This means the model correctly detected 364 out of 369 malignant images.

There were:

```text
5 false-negative predictions
```

A false negative occurs when a malignant image is predicted as benign.

### Benign Recall

The benign recall was approximately:

```text
0.85
```

This means the model correctly classified 149 out of 176 benign images.

There were:

```text
27 false-positive predictions
```

A false positive occurs when a benign image is predicted as malignant.

### Overall F1-score

The macro-average F1-score was:

```text
0.93
```

This indicates good classification performance across the two classes on this test set.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

Move into the project directory:

```bash
cd High_Resolution_Histopathology_AI
```

---

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```bash
pip install torch torchvision streamlit pillow numpy pandas matplotlib scikit-learn
```

---

## ▶️ Running the Streamlit Dashboard

Make sure the following files are present:

```text
app.py
histology_model.pth
```

Run:

```bash
python -m streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

Open this address in your browser.

---

## 🧪 Testing Multiple Images

To test multiple labelled images, arrange them like this:

```text
new_dataset/
├── benign/
│   ├── image1.png
│   └── image2.png
│
└── malignant/
    ├── image3.png
    └── image4.png
```

In `test_many_images.py`, change the test folder path:

```python
test_folder = r"C:\path\to\new_dataset"
```

Then run:

```bash
python test_many_images.py
```

The script will generate:

* Accuracy
* Confusion matrix
* Precision
* Recall
* F1-score
* Detailed CSV results

---

## 🖼️ Testing a Single New Image

For a single new image, update the image path in `predict_new_image.py`:

```python
image_path = r"C:\path\to\new_image.png"
```

Then run:

```bash
python predict_new_image.py
```

The output will show:

```text
Predicted class: BENIGN or MALIGNANT
Confidence: ...
Benign probability: ...
Malignant probability: ...
```

The image must be independent of the training and testing datasets if it is being used as a genuinely new sample.

---

## 🔥 Generating a Heatmap for a New Image

Update the image path in `create_heatmap.py`:

```python
image_path = r"C:\path\to\new_image.png"
```

Run:

```bash
python create_heatmap.py
```

The generated heatmap will be saved in the configured output folder, for example:

```text
outputs/heatmaps/malignancy_heatmap.png
```

---

## ⚠️ Limitations

This project has several limitations:

1. The model was evaluated on a specific dataset and may not perform equally well on images from other sources.
2. Image staining, magnification, resolution, and image quality may affect predictions.
3. The model may produce incorrect predictions.
4. A high confidence score does not guarantee medical correctness.
5. The heatmap shows model response, not confirmed cancer localization.
6. The model should not be used as a standalone medical diagnostic tool.
7. A patient-level dataset split is important to avoid data leakage.
8. More external validation is required before any clinical application.
9. The patch-based heatmap may be less meaningful if all patches receive nearly identical predictions.
10. The current heatmap uses non-overlapping patches, so some image boundaries may not be fully covered.

---

## 🔮 Future Improvements

Possible future improvements include:

* Using patient-level train-validation-test splitting.
* Training with additional histopathology datasets.
* Applying data augmentation.
* Using transfer learning with pretrained weights.
* Testing EfficientNet, DenseNet, or Vision Transformer models.
* Using overlapping patches for better heatmap coverage.
* Adding Grad-CAM or Grad-CAM++ visualization.
* Adding model calibration.
* Adding ROC-AUC and precision-recall curves.
* Adding batch image upload.
* Adding user authentication.
* Deploying the application online.
* Improving the dashboard design.
* Adding a detailed model comparison section.
* Performing external validation on independent datasets.
* Using pathologist-reviewed annotations for localization evaluation.

---

## 💡 Why This Project Is Useful

This project demonstrates how deep learning can be applied to medical image analysis.

It combines:

* Computer vision
* Convolutional neural networks
* Medical image classification
* Model evaluation
* Explainability visualization
* Python programming
* Streamlit application development
* GitHub project management

The dashboard makes the model easier to use because users do not need to run Python prediction commands manually. They can upload an image and immediately view the classification result and patch-level visualization.

---

## 🧑‍💻 Example Workflow

```text
1. Open the Streamlit dashboard.
2. Upload a histopathology image.
3. The image is resized and normalized.
4. ResNet18 analyzes the image.
5. The model predicts benign or malignant.
6. The dashboard displays class probabilities.
7. The image is divided into patches.
8. Each patch is classified separately.
9. Patch scores are combined into a heatmap.
10. The user can inspect or download the results.
```

---

## 📜 Disclaimer

This project is intended for educational, research, and demonstration purposes only.

The predictions generated by this application are not medical diagnoses. The system has not been clinically validated and should not be used to make medical decisions. Any real medical interpretation must be performed by qualified healthcare professionals.

---

## 👤 Author

**Anurag Prasad**

Chemical and Biochemical Engineering
Indian Institute of Technology Patna

---

## ⭐ Acknowledgements

This project uses open-source Python and deep-learning libraries, including:

* PyTorch
* Torchvision
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Pillow

The project is based on the use of breast histopathology image data organized into benign and malignant classes.


