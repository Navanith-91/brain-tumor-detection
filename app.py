import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------
# LOAD MODEL
# --------------------------------
model = tf.keras.models.load_model("brain_tumor_model.h5")

classes = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

# --------------------------------
# DETECT LEVEL
# --------------------------------
def detect_level(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    area = cv2.countNonZero(thresh)

    if area < 2000:
        return "Level 1"
    elif area < 5000:
        return "Level 2"
    elif area < 10000:
        return "Level 3"
    else:
        return "Level 4"

# --------------------------------
# DIET PLAN
# --------------------------------
diet = {
    "Level 1": [
        "Fresh fruits",
        "Leafy vegetables",
        "Whole grains",
        "Drink more water"
    ],
    "Level 2": [
        "High protein meals",
        "Broccoli and spinach",
        "Reduce sugar",
        "Healthy soups"
    ],
    "Level 3": [
        "Soft foods",
        "Frequent small meals",
        "Protein shakes",
        "Hydration support"
    ],
    "Level 4": [
        "Easy-to-swallow meals",
        "High calorie nutrition",
        "Doctor guided diet",
        "Monitor hydration"
    ]
}

# --------------------------------
# PREMIUM CSS
# --------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg,#0f172a,#1e293b);
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}
.title {
    text-align:center;
    font-size:42px;
    font-weight:800;
    color:white;
}
.subtitle {
    text-align:center;
    color:#cbd5e1;
    margin-bottom:25px;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    border:1px solid rgba(255,255,255,0.12);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# HEADER
# --------------------------------
st.markdown('<div class="title">🧠 Brain Tumor Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered MRI Diagnosis + Patient Report</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# --------------------------------
# PATIENT DETAILS
# --------------------------------
st.subheader("👤 Patient Details")

name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=1, max_value=120)
phone = st.text_input("Phone Number")
address = st.text_area("Address")

# --------------------------------
# MRI UPLOAD
# --------------------------------
st.subheader("📤 Upload MRI Scan")

uploaded_file = st.file_uploader(
    "Choose MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))

    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    tumor = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    level = detect_level(image)

    # ----------------------------
    # RESULTS
    # ----------------------------
    st.subheader("🧠 Diagnosis Result")

    if tumor == "No Tumor":
        st.success(f"Tumor Type: {tumor}")
    else:
        st.error(f"Tumor Type: {tumor}")

    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Detected Stage: {level}")

    st.progress(int(confidence))

    # ----------------------------
    # DIET PLAN
    # ----------------------------
    st.subheader("🥗 Recommended Diet Plan")

    for item in diet[level]:
        st.write("✅", item)

    # ----------------------------
    # DOWNLOAD REPORT
    # ----------------------------
    st.subheader("📄 Download Report")

    report = f"""
BRAIN TUMOR DETECTION REPORT
============================

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PATIENT DETAILS
---------------
Name      : {name}
Age       : {age}
Phone     : {phone}
Address   : {address}

DIAGNOSIS
---------
Tumor Type     : {tumor}
Confidence     : {confidence:.2f}%
Detected Level : {level}

DIET PLAN
---------
"""

    for item in diet[level]:
        report += f"- {item}\n"

    report += """

NOTE:
This AI generated result is for educational/demo purposes only.
Consult a medical specialist for clinical diagnosis.
"""

    st.download_button(
        label="⬇ Download Report",
        data=report,
        file_name=f"{name}_BrainTumor_Report.txt",
        mime="text/plain"
    )

st.markdown("</div>", unsafe_allow_html=True)

st.caption("Built with Streamlit + TensorFlow")