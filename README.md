# 🔬 High-Resolution Histopathology AI

> **Deep Learning • Computer Vision • Medical Image Analysis • Explainable AI • Streamlit**

A complete deep-learning pipeline for **breast histopathology image classification** using **ResNet18**, extended with **patch-level analysis, malignancy heatmaps, Grad-CAM explainability, model evaluation, threshold analysis, image-quality statistics, and an interactive Streamlit dashboard**.

The system classifies histopathology images into:

- 🟢 **Benign**
- 🔴 **Malignant**

It is designed as an academic/research project to demonstrate how CNN-based computer vision can be combined with spatial patch analysis and explainability tools for histopathology image analysis.

> ⚠️ **Medical disclaimer:** This project is intended for education, research, and demonstration only. It is **not a clinically validated diagnostic system** and must not be used as a substitute for professional pathological assessment.

---

## ✨ Key Features

### 🧠 Deep Learning Classification
- ResNet18 convolutional neural network
- Transfer learning using ImageNet-pretrained ResNet18 during training
- Binary classification: benign vs malignant
- Softmax-based class probabilities
- Confidence scoring
- CPU/GPU inference support

### 🧩 Patch-Level Analysis
- Automatically divides large histopathology images into patches
- Configurable patch size and stride
- Individual prediction for every patch
- Benign/malignant probability for each patch
- Patch coordinates and dimensions
- Patch brightness and contrast statistics
- Patch uncertainty estimation
- Suspicious-patch threshold analysis

### 🔥 Spatial Malignancy Heatmap
- Converts patch-level malignant probabilities into a spatial probability map
- Overlays the map on the original image
- Supports overlapping patches through configurable stride
- Downloadable heatmap output

### 🔍 Explainable AI
- **Grad-CAM** visualization using the final ResNet18 convolutional block
- Patch coordinate map
- Patch probability visualization
- Helps inspect which image regions influenced model predictions

### 📊 Model Evaluation Center
Using the stored test predictions, the dashboard provides:
- Accuracy
- Precision
- Recall / sensitivity
- Specificity
- F1-score
- Confusion matrix
- ROC curve
- ROC-AUC
- Precision-Recall curve
- Average Precision
- Confidence calibration curve
- Threshold-vs-metric analysis

### 🖥️ Interactive Streamlit Dashboard
- Image upload
- Image metadata
- Image-quality statistics
- Whole-image prediction
- Patch statistics
- Patch gallery/contact sheet
- Heatmap visualization
- Grad-CAM
- Downloadable CSV reports
- Downloadable heatmaps
- Downloadable extracted-patch ZIP
- Analysis summary export

---

## 📌 Project Overview

Histopathology images contain complex cellular and tissue-level patterns. A conventional whole-image classifier produces a single prediction, but that alone does not show how different regions of the image contributed to the result.

This project therefore uses a two-level analysis approach:

**Level 1 — Whole Image**

The complete image is resized to 224 × 224 pixels and passed through ResNet18 to obtain the final benign/malignant prediction.

**Level 2 — Spatial Patch Analysis**

The original high-resolution image is divided into smaller patches. Each patch is independently passed through the same trained model. The resulting malignant probabilities are mapped back to their original spatial locations.

This produces a richer output:

```text
Histopathology Image
        │
        ├──► Whole-image ResNet18 prediction
        │
        └──► Patch extraction
                  │
                  ├──► Patch predictions
                  ├──► Malignant probabilities
                  ├──► Patch statistics
                  ├──► Uncertainty
                  └──► Spatial heatmap
                              │
                              └──► Explainability + visualization
```

---

## 🎯 Objectives

1. Develop a CNN-based benign/malignant histopathology classifier.
2. Apply transfer learning with ResNet18.
3. Evaluate classification performance using multiple metrics.
4. Support inference on previously unseen images.
5. Analyze high-resolution images through patch-based inference.
6. Generate spatial malignancy probability heatmaps.
7. Provide Grad-CAM-based visual explainability.
8. Analyze model behavior across different decision thresholds.
9. Provide an interactive research dashboard using Streamlit.
10. Export prediction and analysis results for further investigation.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **PyTorch** | Deep learning and inference |
| **Torchvision** | ResNet18 and image transformations |
| **ResNet18** | Histopathology image classifier |
| **Pillow (PIL)** | Image loading and processing |
| **NumPy** | Numerical and image operations |
| **Pandas** | Result tables and CSV processing |
| **Matplotlib** | Heatmaps, Grad-CAM and evaluation plots |
| **Scikit-learn** | Classification and evaluation metrics |
| **Streamlit** | Interactive web application |
| **Git/GitHub** | Version control and project hosting |

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────────────┐
                    │   Histopathology Image      │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Image Preprocessing      │
                    │ Resize + Tensor + Normalize  │
                    └──────────────┬──────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
        ┌─────────▼─────────┐             ┌────────▼────────┐
        │ Whole Image       │             │ Patch Extraction │
        │ ResNet18          │             │ Size + Stride    │
        └─────────┬─────────┘             └────────┬────────┘
                  │                                │
        ┌─────────▼─────────┐             ┌────────▼────────┐
        │ Benign/Malignant  │             │ Patch-wise CNN   │
        │ Probability       │             │ Predictions      │
        └─────────┬─────────┘             └────────┬────────┘
                  │                                │
                  │                    ┌───────────┼───────────┐
                  │                    │           │           │
                  │                 Probabilities Statistics Uncertainty
                  │                    │
                  │             ┌──────▼──────────────┐
                  │             │ Spatial Probability │
                  │             │      Heatmap        │
                  │             └──────┬──────────────┘
                  │                    │
                  └──────────┬─────────┘
                             │
                   ┌─────────▼──────────┐
                   │ Streamlit Dashboard│
                   ├────────────────────┤
                   │ Prediction         │
                   │ Patch Explorer      │
                   │ Heatmap             │
                   │ Grad-CAM            │
                   │ Evaluation          │
                   │ Downloads           │
                   └────────────────────┘
```

---

# 📂 Project Structure

```text
High_Resolution_Histopathology_AI/
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── test_many_images.py
│
├── predict_one_image.py
├── predict_patches.py
├── predict_large_image.py
│
├── create_heatmap.py
├── prepare_patches.py
├── extract_training_patches.py
├── visualize_patches.py
├── find_large_image.py
│
├── histology_model.pth
├── test_results.csv
├── requirements.txt
├── README.md
│
├── outputs/
│   ├── heatmaps/
│   │   └── malignancy_heatmap.png
│   ├── patches/
│   └── patch_visualization.png
│
├── patch_dataset/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
│
└── large_histology_image.png
```

---

# 🧠 Model Development

## 1. Dataset

The project uses the **BreaKHis 400X** breast histopathology dataset organized into benign and malignant classes.

The image structure used for model development is:

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

The class mapping stored in the trained checkpoint is:

```python
["benign", "malignant"]
```

The repository also contains the generated patch dataset used in the project workflow.

> For rigorous medical-image evaluation, patient-level separation between training and testing data is important to prevent data leakage.

---

## 2. Image Preprocessing

Input images are converted to RGB and resized to:

```text
224 × 224 pixels
```

The images are normalized using ImageNet statistics:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

The same preprocessing pipeline is used during inference.

---

## 3. ResNet18

The project uses ResNet18 as the main CNN architecture.

During training, an ImageNet-pretrained ResNet18 is loaded and its final fully connected layer is replaced with a two-class classifier:

```text
ResNet18
   ↓
Feature Extraction
   ↓
Fully Connected Layer
   ↓
2 Outputs
   ├── Benign
   └── Malignant
```

The final output logits are converted into class probabilities using Softmax.

```python
probabilities = torch.softmax(output, dim=1)
```

---

## 4. Training Configuration

The current training script uses:

| Parameter | Value |
|---|---:|
| Architecture | ResNet18 |
| Number of classes | 2 |
| Input size | 224 × 224 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Cross Entropy |
| Epochs | 5 |
| Device | CPU/GPU automatically selected |

The trained checkpoint is saved as:

```text
histology_model.pth
```

The checkpoint contains:

- Model state dictionary
- Class names

---

# 🔥 Patch-Based Analysis

A major feature of the project is analysis beyond a single whole-image prediction.

## Patch Extraction

The dashboard supports configurable:

- Patch size
- Patch stride

Default configuration:

```text
Patch size = 224 × 224
Stride     = 224
```

A smaller stride can create overlapping patches and provide denser spatial analysis.

For each patch, the application records:

| Feature | Description |
|---|---|
| Patch number | Unique patch identifier |
| X | Horizontal coordinate |
| Y | Vertical coordinate |
| Width | Patch width |
| Height | Patch height |
| Prediction | Benign or malignant |
| Confidence | Confidence of predicted class |
| Malignant probability | P(malignant) |
| Brightness | Mean image intensity |
| Contrast | Pixel-intensity standard deviation |
| Uncertainty | Probability entropy |

---

## Patch Prediction

Each extracted patch is processed using the same ResNet18 model.

For example:

```text
Patch #12

Prediction: malignant
Confidence: 94.7%
P(benign):  5.3%
P(malignant): 94.7%
```

This makes it possible to inspect which regions of a high-resolution image receive stronger malignant responses.

---

# 🔥 Malignancy Heatmap

The malignant probability from each patch is mapped back to the patch's original coordinates.

Conceptually:

```text
Patch prediction
       ↓
P(malignant)
       ↓
Original patch coordinates
       ↓
Probability matrix
       ↓
Spatial heatmap
       ↓
Overlay on original image
```

The heatmap therefore represents **model-predicted malignant probability**, rather than a confirmed tumor segmentation.

### Interpretation

- Higher values indicate stronger malignant probability from the model.
- Lower values indicate weaker malignant probability.
- The visualization should be interpreted as a model-analysis tool.

---

# 🔍 Grad-CAM Explainability

The application includes Grad-CAM for the ResNet18 model.

Grad-CAM uses gradients flowing into a convolutional feature layer to identify image regions associated with the model's selected prediction.

```text
Input Image
     ↓
ResNet18
     ↓
Convolutional Features
     ↓
Target Class
     ↓
Gradients
     ↓
Grad-CAM
     ↓
Attention Visualization
```

The dashboard can generate a Grad-CAM overlay showing regions that contributed strongly to the model's prediction.

> Grad-CAM is an interpretability visualization, not a ground-truth segmentation or proof of tumor location.

---

# 📊 Model Evaluation

The stored test results contain predictions for **545 test images**.

### Overall Performance

```text
Accuracy = 94.13%
```

### Confusion Matrix

```text
                 Predicted
              Benign  Malignant
Actual Benign    149       27
       Malignant   5      364
```

| Actual | Predicted Benign | Predicted Malignant |
|---|---:|---:|
| Benign | 149 | 27 |
| Malignant | 5 | 364 |

### Classification Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Benign | 0.97 | 0.85 | 0.90 | 176 |
| Malignant | 0.93 | 0.99 | 0.96 | 369 |
| Macro Average | 0.95 | 0.92 | 0.93 | 545 |
| Weighted Average | 0.94 | 0.94 | 0.94 | 545 |

### Important Observation

The malignant class has a recall of approximately **0.99** on this test set.

```text
Malignant samples: 369
Correctly identified: 364
False negatives: 5
```

The benign recall is approximately **0.85**:

```text
Benign samples: 176
Correctly identified: 149
False positives: 27
```

These results are dataset-specific and should not be interpreted as clinical performance.

---

# 📈 Model Evaluation Center

The Streamlit application contains an extended evaluation section based on `test_results.csv`.

## ROC Curve

The ROC curve evaluates the trade-off between:

- True Positive Rate
- False Positive Rate

The dashboard also calculates ROC-AUC.

## Precision-Recall Curve

The Precision-Recall curve shows the relationship between:

- Precision
- Recall

Average Precision is also calculated.

## Threshold Analysis

The dashboard evaluates classification thresholds from:

```text
0.05 → 0.95
```

Instead of assuming that 0.50 is the only possible threshold, the application calculates:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score

for multiple thresholds.

The threshold producing the highest F1-score can be identified for research analysis.

> Threshold selection should be performed on a dedicated validation set rather than tuned on the final test set for formal model evaluation.

## Confidence Calibration

The dashboard also provides a calibration plot comparing predicted probabilities with observed malignant frequencies.

This helps investigate whether a confidence such as 0.90 corresponds reasonably well to an approximately 90% observed frequency.

---

# 🖼️ Image Quality Analysis

Before/alongside prediction, the dashboard calculates simple image statistics:

- Brightness
- Contrast
- Sharpness proxy
- Background fraction
- Tissue fraction

These measurements help identify potential differences in image quality and tissue coverage.

They are descriptive statistics and are **not medical quality-control measurements**.

---

# 📦 Batch Inference

The dashboard can analyze multiple uploaded images in a single session.

For each image, the analysis can report:

- Image name
- Dimensions
- Prediction
- Confidence
- Benign probability
- Malignant probability
- Patch count
- Mean malignant probability
- Median malignant probability
- Maximum malignant probability
- Malignant patch fraction
- Suspicious patch fraction

Results can be exported as CSV.

---

# 📥 Downloadable Outputs

The application supports several research-oriented downloads.

### Patch Results CSV

Contains patch-level information including:

```text
Patch ID
Coordinates
Dimensions
Prediction
Confidence
Malignant Probability
Brightness
Contrast
```

### Heatmap

Download the generated spatial malignancy heatmap as a PNG image.

### Extracted Patches

All extracted patches can be downloaded as a ZIP archive.

### Analysis Summary

A compact CSV containing image-level and patch-level summary statistics.

---

# 🖥️ Streamlit Dashboard

The application is organized around an interactive dashboard.

### Dashboard Workflow

```text
1. Upload image
       ↓
2. Read image metadata
       ↓
3. Whole-image ResNet18 prediction
       ↓
4. Calculate class probabilities
       ↓
5. Extract patches
       ↓
6. Predict every patch
       ↓
7. Calculate patch statistics
       ↓
8. Generate heatmap
       ↓
9. Generate optional Grad-CAM
       ↓
10. Explore patches
       ↓
11. Download analysis outputs
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/anurag3401/High_Resolution_Histopathology_AI.git
cd High_Resolution_Histopathology_AI
```

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

Make sure the following files are available in the repository root:

```text
app.py
histology_model.pth
test_results.csv
```

Run:

```bash
python -m streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# 🧪 Run Individual Scripts

## Train the Model

Update the dataset paths in `train_model.py`, then:

```bash
python train_model.py
```

Output:

```text
histology_model.pth
```

## Evaluate the Model

```bash
python evaluate_model.py
```

## Test Many Images

Update the model, dataset, and output paths in `test_many_images.py`:

```bash
python test_many_images.py
```

This generates detailed predictions and:

```text
test_results.csv
```

## Predict One Image

Update the input image path in the corresponding prediction script and run:

```bash
python predict_one_image.py
```

## Predict a Large Image

```bash
python predict_large_image.py
```

## Generate Heatmap

```bash
python create_heatmap.py
```

---

# 📋 Main Scripts

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit dashboard |
| `train_model.py` | Train ResNet18 classifier |
| `evaluate_model.py` | Evaluate model on test dataset |
| `test_many_images.py` | Run batch image evaluation and save CSV results |
| `predict_one_image.py` | Single-image inference |
| `predict_patches.py` | Patch-level prediction |
| `predict_large_image.py` | Large-image analysis |
| `create_heatmap.py` | Generate patch-based heatmap |
| `prepare_patches.py` | Prepare image patches |
| `extract_training_patches.py` | Extract patches for training |
| `visualize_patches.py` | Visualize extracted patches |
| `find_large_image.py` | Locate large histopathology images |

---

# ⚙️ Important Configuration

The Streamlit sidebar allows the user to configure:

### Patch Size

Available options include:

```text
224
256
280
320
```

The recommended default is **224 × 224**, matching the model's training input size.

### Patch Stride

Available options include:

```text
112
168
224
```

Smaller stride → more overlap → denser analysis → longer inference time.

### Suspicious Patch Threshold

The dashboard allows a threshold to be selected for identifying patches with higher malignant probability.

This threshold is intended for **research visualization**, not clinical decision-making.

---

# ⚠️ Limitations

Despite the extended analysis capabilities, several limitations remain:

1. The model is trained and evaluated on a specific dataset distribution.
2. Performance may change on images from different laboratories, scanners, staining protocols, magnifications, or populations.
3. The reported 94.13% accuracy is specific to the evaluated test set.
4. Dataset imbalance affects some metrics.
5. The model can produce incorrect predictions.
6. High confidence does not guarantee correctness.
7. Patch predictions do not constitute tumor segmentation.
8. Grad-CAM does not provide ground-truth pathological localization.
9. The current training workflow should be improved with patient-level splitting and stronger validation.
10. External validation on independent datasets is required for meaningful generalization assessment.
11. Threshold optimization should be performed on a validation set rather than the final test set.
12. Image-quality statistics are simple computational proxies rather than clinically validated measurements.
13. CPU inference can become slow when using many overlapping patches.
14. No clinical workflow, patient records, or medical decision support is implemented.

---

# 🔮 Future Improvements

Potential extensions include:

- Patient-level train/validation/test splitting
- Stronger data augmentation
- Stratified validation
- K-fold cross-validation
- External dataset validation
- EfficientNet/DenseNet comparison
- Vision Transformer comparison
- Ensemble models
- Better class-imbalance handling
- Probability calibration using a validation set
- Validation-based operating-point selection
- Overlapping multi-scale patch analysis
- Tissue/background segmentation before patch inference
- Pathologist-reviewed region annotations
- Quantitative localization evaluation
- Segmentation models such as U-Net
- Attention-based multiple-instance learning
- Whole-slide image support
- Experiment tracking
- Model versioning
- Automated testing and CI/CD
- Cloud deployment
- User authentication
- Improved inference optimization

---

# 💡 What Makes This Project More Than a Basic CNN Classifier?

A conventional image-classification project may only provide:

```text
Image → CNN → Class
```

This project extends that workflow into:

```text
Image
  ↓
CNN Classification
  ↓
Class Probabilities
  ↓
Patch Extraction
  ↓
Patch-wise Classification
  ↓
Spatial Probability Mapping
  ↓
Heatmap
  ↓
Grad-CAM Explainability
  ↓
Uncertainty + Image Statistics
  ↓
Evaluation + Threshold Analysis
  ↓
Downloadable Research Outputs
```

This makes the project a broader **computer-vision and explainable-AI application** rather than only a binary classification model.

---

# 📊 Project Results at a Glance

| Metric | Result |
|---|---:|
| Model | ResNet18 |
| Task | Binary classification |
| Classes | Benign / Malignant |
| Input size | 224 × 224 |
| Test images | 545 |
| Accuracy | **94.13%** |
| Malignant recall | **0.99** |
| Malignant F1-score | **0.96** |
| Macro F1-score | **0.93** |
| Patch analysis | ✅ |
| Heatmap | ✅ |
| Grad-CAM | ✅ |
| ROC / PR analysis | ✅ |
| Threshold analysis | ✅ |
| Calibration analysis | ✅ |
| Batch inference | ✅ |
| CSV export | ✅ |
| Patch ZIP export | ✅ |
| Streamlit dashboard | ✅ |

---

# 🧑‍💻 Author

### **Anurag Prasad**

**Department of Chemical & Biochemical Engineering**  
**Indian Institute of Technology Patna**

GitHub: [@anurag3401](https://github.com/anurag3401)

---

# 📜 Disclaimer

This project is developed for **academic, educational, research, and demonstration purposes**.

The predictions, confidence values, patch probabilities, heatmaps, Grad-CAM visualizations, and other outputs are generated by a machine-learning model and **do not constitute a medical diagnosis**.

The system has not been clinically validated and should not be used to make patient-care decisions. Any medical interpretation must be performed by qualified healthcare professionals.

---

# ⭐ Acknowledgements

This project uses open-source tools and libraries including:

- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow

The project uses breast histopathology image data organized into benign and malignant classes and demonstrates deep-learning-based medical image analysis in an academic setting.

---

## ⭐ If you find this project useful, consider giving the repository a star!
