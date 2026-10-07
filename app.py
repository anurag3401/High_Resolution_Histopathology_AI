import os
import io
import time
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image, ImageDraw
from torchvision import models, transforms
import matplotlib.pyplot as plt


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
        .main { background-color: #f5f7fb; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        .hero {
            padding: 1.5rem 2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #172554 0%, #1e3a8a 50%, #2563eb 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 25px rgba(30, 58, 138, 0.25);
        }
        .hero h1 { font-size: 2.35rem; margin-bottom: .25rem; }
        .hero p { font-size: 1.03rem; opacity: .9; }

        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 15px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 12px rgba(15, 23, 42, .05);
        }
        .metric-title { color: #64748b; font-size: .85rem; }
        .metric-value { color: #0f172a; font-size: 1.55rem; font-weight: 700; }

        .section-title {
            color: #0f172a;
            font-size: 1.3rem;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: .75rem;
        }

        .result-box {
            padding: 1.4rem;
            border-radius: 18px;
            text-align: center;
            margin-top: .7rem;
        }

        .warning-box {
            background-color: #fff7ed;
            border-left: 5px solid #f97316;
            padding: 1rem;
            border-radius: 10px;
            color: #7c2d12;
        }

        [data-testid="stSidebar"] { background-color: #eef2ff; }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "histology_model.pth")

MODEL_INPUT_SIZE = 224
DEFAULT_PATCH_SIZE = 224
DEFAULT_STRIDE = 224


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    class_names = checkpoint.get("classes", ["benign", "malignant"])
    model.eval()

    return model, class_names, device


transform = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION HELPERS
# ============================================================

def predict_image(image, model, class_names, device):
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()

    probability_dict = {
        name: probabilities[0][i].item()
        for i, name in enumerate(class_names)
    }

    return class_names[predicted_index], probabilities[0][predicted_index].item(), probability_dict


def analyze_patches(
    image,
    model,
    class_names,
    device,
    patch_size=DEFAULT_PATCH_SIZE,
    stride=DEFAULT_STRIDE
):
    """
    Extract every valid patch and run ResNet18 on each patch.
    Returns patch metadata, actual PIL patches, and an averaged
    malignant-probability heatmap.
    """

    image_width, image_height = image.size
    image_array = np.asarray(image)

    heatmap = np.zeros((image_height, image_width), dtype=np.float32)
    count_map = np.zeros((image_height, image_width), dtype=np.float32)

    malignant_index = (
        class_names.index("malignant")
        if "malignant" in class_names
        else 1
    )

    records = []

    start = time.perf_counter()

    with torch.no_grad():
        patch_number = 0

        # Include edge patches by stepping across the complete image.
        for y in range(0, image_height, stride):
            for x in range(0, image_width, stride):

                right = min(x + patch_size, image_width)
                bottom = min(y + patch_size, image_height)

                patch = image.crop((x, y, right, bottom))

                if patch.width < 32 or patch.height < 32:
                    continue

                tensor = transform(patch).unsqueeze(0).to(device)
                probabilities = torch.softmax(model(tensor), dim=1)

                predicted_index = torch.argmax(probabilities, dim=1).item()
                malignant_probability = probabilities[0][malignant_index].item()
                confidence = probabilities[0][predicted_index].item()

                # Basic patch image statistics.
                arr = np.asarray(patch).astype(np.float32)
                brightness = float(arr.mean())
                contrast = float(arr.std())

                records.append({
                    "patch_number": patch_number + 1,
                    "x": x,
                    "y": y,
                    "width": patch.width,
                    "height": patch.height,
                    "prediction": class_names[predicted_index],
                    "confidence": confidence,
                    "malignant_probability": malignant_probability,
                    "brightness": brightness,
                    "contrast": contrast,
                    "_image": patch.copy()
                })

                heatmap[y:bottom, x:right] += malignant_probability
                count_map[y:bottom, x:right] += 1

                patch_number += 1

    count_map[count_map == 0] = 1
    heatmap = heatmap / count_map

    elapsed = time.perf_counter() - start

    return records, heatmap, elapsed


def make_heatmap_figure(image, heatmap):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(image)
    overlay = ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.48,
        vmin=0,
        vmax=1
    )
    ax.axis("off")
    ax.set_title("Patch-Based Malignancy Heatmap", fontsize=15, fontweight="bold")

    colorbar = fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Malignant probability")
    fig.tight_layout()

    return fig


def make_patch_contact_sheet(records, columns=4, cell_size=180):
    """Create a single visual grid containing extracted patches."""
    if not records:
        return None

    rows = int(np.ceil(len(records) / columns))
    sheet = Image.new("RGB", (columns * cell_size, rows * (cell_size + 30)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, record in enumerate(records):
        patch = record["_image"].copy()
        patch.thumbnail((cell_size - 8, cell_size - 8))

        x0 = (i % columns) * cell_size
        y0 = (i // columns) * (cell_size + 30)

        px = x0 + (cell_size - patch.width) // 2
        py = y0 + 4

        sheet.paste(patch, (px, py))

        label = (
            f"#{record['patch_number']}  "
            f"{record['prediction']}  "
            f"{record['malignant_probability'] * 100:.0f}%"
        )
        draw.text((x0 + 5, y0 + cell_size + 3), label, fill="black")

    return sheet


def records_to_dataframe(records):
    columns = [
        "patch_number",
        "x",
        "y",
        "width",
        "height",
        "prediction",
        "confidence",
        "malignant_probability",
        "brightness",
        "contrast"
    ]

    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(columns=columns)

    return df[columns].copy()


def create_patch_zip(records):
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for record in records:
            patch_buffer = io.BytesIO()
            record["_image"].save(patch_buffer, format="PNG")

            name = (
                f"patch_{record['patch_number']:03d}_"
                f"{record['prediction']}_"
                f"{record['malignant_probability'] * 100:.1f}pct.png"
            )

            zf.writestr(name, patch_buffer.getvalue())

    buffer.seek(0)
    return buffer


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔬 Histology AI Dashboard</h1>
        <p>
            ResNet18-based benign vs malignant classification with
            patch-level analysis, probability mapping, patch explorer,
            image statistics and downloadable research outputs.
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

    st.divider()

    st.subheader("Patch Analysis")

    patch_size = st.selectbox(
        "Patch size",
        [224, 256, 280, 320],
        index=0,
        help="The original 224×224 setting is recommended because the model was trained using 224×224 inputs."
    )

    stride = st.selectbox(
        "Patch stride",
        [112, 168, 224],
        index=2,
        help="Smaller stride creates overlapping patches and a denser heatmap, but increases inference time."
    )

    show_heatmap = st.checkbox("Generate heatmap", value=True)
    show_all_patches = st.checkbox("Show all extracted patches", value=True)

    max_gallery = st.slider(
        "Maximum patches in gallery",
        min_value=4,
        max_value=100,
        value=24,
        step=4
    )

    suspicious_threshold = st.slider(
        "Suspicious patch threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05
    )

    st.divider()

    st.caption(f"Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    st.warning(
        "Academic/research use only. This system is not a medical diagnostic tool."
    )


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error("Model file was not found. Make sure histology_model.pth is in the repository root.")
    st.stop()

try:
    model, class_names, device = load_model()
except Exception as error:
    st.error("Error while loading the model.")
    st.exception(error)
    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload Histopathology Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PNG, JPG, JPEG, BMP, TIF or TIFF image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is None:
    st.info("Upload an image to start the full patch-level analysis.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
width, height = image.size


# ============================================================
# IMAGE PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">📐 Image & Model Parameters</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

metrics = [
    ("Image Size", f"{width} × {height}"),
    ("Pixels", f"{width * height:,}"),
    ("Aspect Ratio", f"{width / height:.2f}"),
    ("Patch Size", f"{patch_size} px"),
    ("Stride", f"{stride} px")
]

for column, (title, value) in zip([c1, c2, c3, c4, c5], metrics):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
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

st.image(image, caption=uploaded_file.name, use_container_width=True)


# ============================================================
# IMAGE-LEVEL PREDICTION
# ============================================================

with st.spinner("Running image-level ResNet18 prediction..."):
    image_start = time.perf_counter()
    predicted_class, confidence, probability_dict = predict_image(
        image, model, class_names, device
    )
    image_time = time.perf_counter() - image_start

if predicted_class.lower() == "malignant":
    result_background = "#fee2e2"
    result_text = "#991b1b"
    result_icon = "⚠️"
else:
    result_background = "#dcfce7"
    result_text = "#166534"
    result_icon = "✅"

st.markdown(
    '<div class="section-title">🧠 Image-Level Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="result-box"
         style="background-color:{result_background}; color:{result_text};">
        <div style="font-size:2.3rem;">{result_icon}</div>
        <div style="font-size:1rem;">Predicted Class</div>
        <div style="font-size:2.15rem;font-weight:700;">{predicted_class.upper()}</div>
        <div style="font-size:1.05rem;">Confidence: {confidence * 100:.2f}%</div>
    </div>
    """,
    unsafe_allow_html=True
)

p1, p2, p3 = st.columns(3)

with p1:
    st.metric("Benign probability", f"{probability_dict.get('benign', 0) * 100:.2f}%")
with p2:
    st.metric("Malignant probability", f"{probability_dict.get('malignant', 0) * 100:.2f}%")
with p3:
    st.metric("Image inference time", f"{image_time:.3f} s")


# ============================================================
# PATCH ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🧩 Patch-Level Analysis</div>',
    unsafe_allow_html=True
)

with st.spinner("Extracting patches and running patch-level inference..."):
    records, heatmap, patch_time = analyze_patches(
        image=image,
        model=model,
        class_names=class_names,
        device=device,
        patch_size=patch_size,
        stride=stride
    )

patch_df = records_to_dataframe(records)

if patch_df.empty:
    st.error("No valid patches could be extracted from this image.")
    st.stop()


# ============================================================
# PATCH SUMMARY
# ============================================================

total_patches = len(patch_df)
malignant_count = int(
    patch_df["prediction"].str.lower().eq("malignant").sum()
)
benign_count = total_patches - malignant_count

mean_malignant = float(patch_df["malignant_probability"].mean())
median_malignant = float(patch_df["malignant_probability"].median())
max_malignant = float(patch_df["malignant_probability"].max())
min_malignant = float(patch_df["malignant_probability"].min())
std_malignant = float(patch_df["malignant_probability"].std(ddof=0))
suspicious_count = int(
    (patch_df["malignant_probability"] >= suspicious_threshold).sum()
)

malignant_fraction = malignant_count / total_patches
suspicious_fraction = suspicious_count / total_patches

s1, s2, s3, s4, s5, s6 = st.columns(6)

summary = [
    ("Total patches", total_patches),
    ("Malignant patches", malignant_count),
    ("Benign patches", benign_count),
    ("Mean malignant P", f"{mean_malignant * 100:.1f}%"),
    ("Max malignant P", f"{max_malignant * 100:.1f}%"),
    ("Suspicious patches", suspicious_count)
]

for column, (title, value) in zip([s1, s2, s3, s4, s5, s6], summary):
    with column:
        st.metric(title, value)


st.caption(
    f"Patch inference time: {patch_time:.3f} s | "
    f"Patch density: {total_patches / max(width * height / 1_000_000, 1e-9):.1f} patches/MP"
)


# ============================================================
# PATCH DISTRIBUTION
# ============================================================

left, right = st.columns(2)

with left:
    st.subheader("Patch Prediction Distribution")
    distribution = pd.DataFrame({
        "Class": ["Benign", "Malignant"],
        "Patches": [benign_count, malignant_count]
    }).set_index("Class")
    st.bar_chart(distribution)

with right:
    st.subheader("Malignant Probability Distribution")
    probability_bins = pd.cut(
        patch_df["malignant_probability"],
        bins=[0, .2, .4, .6, .8, 1.0],
        labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
        include_lowest=True
    )
    histogram = probability_bins.value_counts().sort_index()
    st.bar_chart(histogram)


# ============================================================
# PATCH TABLE
# ============================================================

st.subheader("📋 Patch Prediction Table")

display_df = patch_df.copy()
display_df["confidence"] = (display_df["confidence"] * 100).round(2)
display_df["malignant_probability"] = (
    display_df["malignant_probability"] * 100
).round(2)
display_df["brightness"] = display_df["brightness"].round(2)
display_df["contrast"] = display_df["contrast"].round(2)

display_df = display_df.rename(columns={
    "patch_number": "Patch",
    "x": "X",
    "y": "Y",
    "width": "Width",
    "height": "Height",
    "prediction": "Prediction",
    "confidence": "Confidence (%)",
    "malignant_probability": "Malignant P (%)",
    "brightness": "Brightness",
    "contrast": "Contrast"
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SUSPICIOUS PATCHES
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Highest-Risk / Most Suspicious Patches</div>',
    unsafe_allow_html=True
)

top_n = min(8, total_patches)

top_records = sorted(
    records,
    key=lambda r: r["malignant_probability"],
    reverse=True
)[:top_n]

top_cols = st.columns(min(4, top_n))

for i, record in enumerate(top_records):
    with top_cols[i % len(top_cols)]:
        st.image(
            record["_image"],
            use_container_width=True,
            caption=(
                f"Patch #{record['patch_number']} | "
                f"{record['prediction']} | "
                f"Malignant P = {record['malignant_probability'] * 100:.1f}%"
            )
        )


# ============================================================
# ALL PATCHES / CONTACT SHEET
# ============================================================

if show_all_patches:
    st.markdown(
        '<div class="section-title">🧱 Extracted Patch Explorer</div>',
        unsafe_allow_html=True
    )

    gallery_records = sorted(
        records,
        key=lambda r: r["patch_number"]
    )[:max_gallery]

    st.caption(
        f"Showing {len(gallery_records)} of {total_patches} extracted patches. "
        f"Increase the gallery limit from the sidebar to view more."
    )

    # Display actual individual patches in a responsive grid.
    gallery_cols = st.columns(4)

    for i, record in enumerate(gallery_records):
        with gallery_cols[i % 4]:
            st.image(
                record["_image"],
                use_container_width=True,
                caption=(
                    f"#{record['patch_number']} • "
                    f"{record['prediction']} • "
                    f"Malignant {record['malignant_probability'] * 100:.1f}%"
                )
            )

    with st.expander("View compact patch contact sheet"):
        contact_sheet = make_patch_contact_sheet(gallery_records, columns=4)
        st.image(contact_sheet, use_container_width=True)


# ============================================================
# HEATMAP
# ============================================================

if show_heatmap:
    st.markdown(
        '<div class="section-title">🔥 Spatial Malignancy Heatmap</div>',
        unsafe_allow_html=True
    )

    heatmap_figure = make_heatmap_figure(image, heatmap)

    image_col, heatmap_col = st.columns(2)

    with image_col:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with heatmap_col:
        st.subheader("Patch Probability Overlay")
        st.pyplot(heatmap_figure, use_container_width=True)

    heatmap_buffer = io.BytesIO()
    heatmap_figure.savefig(
        heatmap_buffer,
        format="png",
        dpi=200,
        bbox_inches="tight"
    )
    heatmap_buffer.seek(0)

    st.download_button(
        "⬇️ Download Heatmap",
        data=heatmap_buffer,
        file_name="malignancy_heatmap.png",
        mime="image/png"
    )


# ============================================================
# DOWNLOADABLE RESEARCH OUTPUTS
# ============================================================

st.markdown(
    '<div class="section-title">📥 Download Analysis Outputs</div>',
    unsafe_allow_html=True
)

download_col1, download_col2, download_col3 = st.columns(3)

csv_buffer = display_df.to_csv(index=False).encode("utf-8")

with download_col1:
    st.download_button(
        "⬇️ Patch Results CSV",
        data=csv_buffer,
        file_name="patch_predictions.csv",
        mime="text/csv"
    )

with download_col2:
    patch_zip = create_patch_zip(records)
    st.download_button(
        "⬇️ Download All Patches (ZIP)",
        data=patch_zip,
        file_name="extracted_patches.zip",
        mime="application/zip"
    )

with download_col3:
    report = pd.DataFrame([{
        "image": uploaded_file.name,
        "width": width,
        "height": height,
        "patch_size": patch_size,
        "stride": stride,
        "total_patches": total_patches,
        "image_prediction": predicted_class,
        "image_confidence": confidence,
        "mean_malignant_probability": mean_malignant,
        "median_malignant_probability": median_malignant,
        "max_malignant_probability": max_malignant,
        "malignant_patch_fraction": malignant_fraction,
        "suspicious_patch_fraction": suspicious_fraction,
        "suspicious_threshold": suspicious_threshold,
        "patch_inference_seconds": patch_time
    }])

    st.download_button(
        "⬇️ Analysis Summary CSV",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="analysis_summary.csv",
        mime="text/csv"
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">📈 Research Interpretation</div>',
    unsafe_allow_html=True
)

st.info(
    f"""
    **Patch coverage summary:** {malignant_count}/{total_patches}
    ({malignant_fraction * 100:.1f}%) of patches were classified as malignant.

    **Probability summary:** mean malignant probability =
    {mean_malignant * 100:.1f}%, median =
    {median_malignant * 100:.1f}%, maximum =
    {max_malignant * 100:.1f}%.

    **Threshold analysis:** {suspicious_count} patches
    ({suspicious_fraction * 100:.1f}%) reached the selected
    suspicious threshold of {suspicious_threshold * 100:.0f}%.
    """
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="warning-box">
        <b>Important:</b>
        This model provides an academic image-classification and
        patch-analysis result. The prediction, patch scores and heatmap
        are not a medical diagnosis. Results can be affected by staining,
        magnification, image quality, dataset bias and distribution shift.
        Expert pathological review is required for any medical application.
    </div>
    """,
    unsafe_allow_html=True
)
