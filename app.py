import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Histology AI Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f5f7fb;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.5rem 2rem;
            border-radius: 20px;
            background: linear-gradient(
                135deg,
                #172554 0%,
                #1e3a8a 50%,
                #2563eb 100%
            );
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 25px rgba(30, 58, 138, 0.25);
        }

        .hero h1 {
            font-size: 2.4rem;
            margin-bottom: 0.3rem;
        }

        .hero p {
            font-size: 1.05rem;
            opacity: 0.9;
        }

        .metric-card {
            background: white;
            padding: 1.2rem;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        }

        .metric-title {
            color: #64748b;
            font-size: 0.9rem;
        }

        .metric-value {
            color: #0f172a;
            font-size: 1.8rem;
            font-weight: 700;
        }

        .section-title {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }

        .result-box {
            padding: 1.5rem;
            border-radius: 18px;
            text-align: center;
            margin-top: 1rem;
        }

        .warning-box {
            background-color: #fff7ed;
            border-left: 5px solid #f97316;
            padding: 1rem;
            border-radius: 10px;
            color: #7c2d12;
        }

        .info-box {
            background-color: #eff6ff;
            border-left: 5px solid #3b82f6;
            padding: 1rem;
            border-radius: 10px;
            color: #1e3a8a;
        }

        [data-testid="stSidebar"] {
            background-color: #eef2ff;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS AND SETTINGS
# ============================================================

MODEL_PATH = (
    r"C:\\Users\\Anurag Prasad\\Desktop\\High_Resolution_Histopathology_AI"
    r"\\histology_model.pth"
)

PATCH_SIZE = 224
STRIDE = 224


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    class_names = checkpoint["classes"]

    model.eval()

    return model, class_names, device


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

def predict_image(image, model, class_names, device):

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        predicted_class = class_names[predicted_index]
        confidence = probabilities[0][predicted_index].item()

    probability_dict = {
        class_name: probabilities[0][index].item()
        for index, class_name in enumerate(class_names)
    }

    return predicted_class, confidence, probability_dict


# ============================================================
# PATCH HEATMAP
# ============================================================

def create_heatmap(image, model, class_names, device):

    image_array = np.array(image)

    image_width, image_height = image.size

    heatmap = np.zeros(
        (image_height, image_width),
        dtype=np.float32
    )

    count_map = np.zeros(
        (image_height, image_width),
        dtype=np.float32
    )

    malignant_index = class_names.index("malignant")

    patch_records = []

    with torch.no_grad():

        for y in range(0, image_height, STRIDE):

            for x in range(0, image_width, STRIDE):

                right = min(
                    x + PATCH_SIZE,
                    image_width
                )

                bottom = min(
                    y + PATCH_SIZE,
                    image_height
                )

                patch = image.crop(
                    (x, y, right, bottom)
                )

                # Ignore very small edge patches
                if patch.width < 32 or patch.height < 32:
                    continue

                patch_tensor = transform(patch)
                patch_tensor = patch_tensor.unsqueeze(0)
                patch_tensor = patch_tensor.to(device)

                output = model(patch_tensor)
                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                malignant_probability = (
                    probabilities[0][malignant_index].item()
                )

                heatmap[y:bottom, x:right] += malignant_probability
                count_map[y:bottom, x:right] += 1

                predicted_index = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

                patch_records.append({
                    "x": x,
                    "y": y,
                    "predicted_class": class_names[predicted_index],
                    "malignant_probability": malignant_probability
                })

    count_map[count_map == 0] = 1

    heatmap = heatmap / count_map

    return heatmap, patch_records


# ============================================================
# HEATMAP FIGURE
# ============================================================

def make_heatmap_figure(image, heatmap):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.imshow(image)

    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.48,
        vmin=0,
        vmax=1
    )

    ax.axis("off")
    ax.set_title(
        "Patch-Based Malignancy Heatmap",
        fontsize=15,
        fontweight="bold"
    )

    colorbar = fig.colorbar(
        ax.images[-1],
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    colorbar.set_label(
        "Malignant probability"
    )

    fig.tight_layout()

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔬 Histology AI Dashboard</h1>
        <p>
            ResNet18-based benign and malignant histopathology
            image classification with patch-level heatmap analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Dashboard Settings")

    st.write("**Model:** ResNet18")
    st.write("**Task:** Benign vs Malignant")
    st.write("**Input:** Histopathology image")
    st.write("**Device:** CPU/GPU automatically selected")

    st.divider()

    st.subheader("Heatmap Settings")

    show_heatmap = st.checkbox(
        "Generate patch heatmap",
        value=True
    )

    st.divider()

    st.warning(
        "This dashboard is for academic and research use only. "
        "It is not a medical diagnostic tool."
    )


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model file was not found. Check MODEL_PATH in app.py."
    )

    st.stop()

try:

    model, class_names, device = load_model()

except Exception as error:

    st.error("Error while loading the model:")
    st.exception(error)
    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload a New Histopathology Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)


if uploaded_file is None:

    st.info(
        "Upload a new image to view the prediction and heatmap."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

image = Image.open(uploaded_file).convert("RGB")


# ============================================================
# IMAGE INFORMATION CARDS
# ============================================================

width, height = image.size

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Image Width</div>
            <div class="metric-value">{width}px</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Image Height</div>
            <div class="metric-value">{height}px</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Patch Size</div>
            <div class="metric-value">224</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Device</div>
            <div class="metric-value">{str(device).upper()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ORIGINAL IMAGE
# ============================================================

st.markdown(
    '<div class="section-title">🖼️ Uploaded Image</div>',
    unsafe_allow_html=True
)

st.image(
    image,
    caption=uploaded_file.name,
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

with st.spinner("Analyzing image..."):

    predicted_class, confidence, probability_dict = predict_image(
        image,
        model,
        class_names,
        device
    )


st.markdown(
    '<div class="section-title">🧠 Model Prediction</div>',
    unsafe_allow_html=True
)

if predicted_class.lower() == "malignant":

    result_background = "#fee2e2"
    result_text = "#991b1b"
    result_icon = "⚠️"

else:

    result_background = "#dcfce7"
    result_text = "#166534"
    result_icon = "✅"


st.markdown(
    f"""
    <div class="result-box"
         style="background-color:{result_background};
                color:{result_text};">

        <div style="font-size:2.5rem;">{result_icon}</div>

        <div style="font-size:1.1rem;">
            Predicted Class
        </div>

        <div style="font-size:2.2rem;font-weight:700;">
            {predicted_class.upper()}
        </div>

        <div style="font-size:1.1rem;">
            Model confidence: {confidence * 100:.2f}%
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROBABILITY DISPLAY
# ============================================================

st.markdown(
    '<div class="section-title">📊 Class Probabilities</div>',
    unsafe_allow_html=True
)

probability_df = pd.DataFrame({
    "Class": list(probability_dict.keys()),
    "Probability": [
        value * 100
        for value in probability_dict.values()
    ]
})

st.dataframe(
    probability_df,
    use_container_width=True,
    hide_index=True
)

for class_name, probability in probability_dict.items():

    st.write(
        f"**{class_name.capitalize()}**: "
        f"{probability * 100:.2f}%"
    )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )


# ============================================================
# HEATMAP
# ============================================================

if show_heatmap:

    st.markdown(
        '<div class="section-title">🔥 Patch-Based Heatmap</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Generating heatmap..."):

        heatmap, patch_records = create_heatmap(
            image,
            model,
            class_names,
            device
        )

        heatmap_figure = make_heatmap_figure(
            image,
            heatmap
        )

    image_col, heatmap_col = st.columns(2)

    with image_col:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with heatmap_col:

        st.subheader("Malignancy Heatmap")

        st.pyplot(
            heatmap_figure,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Patch statistics
    # --------------------------------------------------------

    patch_df = pd.DataFrame(patch_records)

    if not patch_df.empty:

        total_patches = len(patch_df)

        malignant_patches = (
            patch_df["predicted_class"]
            .str.lower()
            .eq("malignant")
            .sum()
        )

        average_probability = (
            patch_df["malignant_probability"]
            .mean()
        )

        highest_probability = (
            patch_df["malignant_probability"]
            .max()
        )

        st.markdown(
            '<div class="section-title">📌 Patch Statistics</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Total Patches",
                total_patches
            )

        with c2:

            st.metric(
                "Malignant Patches",
                malignant_patches
            )

        with c3:

            st.metric(
                "Highest Malignant Probability",
                f"{highest_probability * 100:.2f}%"
            )

        st.dataframe(
            patch_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Download heatmap
        # ----------------------------------------------------

        heatmap_buffer = io.BytesIO()

        heatmap_figure.savefig(
            heatmap_buffer,
            format="png",
            dpi=200,
            bbox_inches="tight"
        )

        heatmap_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Heatmap",
            data=heatmap_buffer,
            file_name="malignancy_heatmap.png",
            mime="image/png"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="warning-box">
        <b>Important:</b>
        This model provides an academic classification result.
        The prediction and heatmap are not a medical diagnosis.
        Results may be affected by image quality, staining,
        magnification, dataset bias, and training-data leakage.
        Always require expert pathological review for medical use.
    </div>
    """,
    unsafe_allow_html=True
)