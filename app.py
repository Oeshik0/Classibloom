import os
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model

# App Configuration
st.set_page_config(
    page_title="Bloom Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 24px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #45a049;
}
.stFileUploader>div>div>button {
    background-color: #4CAF50;
    color: white;
}
.prediction-box {
    padding: 25px;
    border-radius: 12px;
    background: linear-gradient(145deg, #e0f2f1, #ffffff);
    box-shadow: 4px 4px 10px #ccc;
    margin-top: 25px;
}
h2 {
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h2 class='header-text'>🌸 Bloom Classification CNN Model 🌸</h2>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#666; font-size:16px;'>
Upload a flower image to classify it as one of these: 
<b>daisy, dandelion, rose, sunflower, or tulip</b>.
</p>
""", unsafe_allow_html=True)

# Flower classes and emojis
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
flower_emojis = {
    'daisy': '🌼',
    'dandelion': '🌱',
    'rose': '🌹',
    'sunflower': '🌻',
    'tulip': '🌷'
}

# Load model
@st.cache_resource
def load_flower_model():
    return load_model('Flower_Recog_Model.h5')

model = load_flower_model()

# Image classification logic
def classify_image(image_path):
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        predicted_class = flower_names[np.argmax(result)]
        confidence = np.max(result) * 100

        top3_indices = np.argsort(result)[::-1][:3]
        top3_classes = [flower_names[i] for i in top3_indices]
        top3_confidences = [result[i] * 100 for i in top3_indices]

        return predicted_class, confidence, top3_classes, top3_confidences
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None

# Upload section in tabs
tab1, tab2 = st.tabs(["🌸 Upload & Predict", "📊 Prediction Chart"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a flower image...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Flower Image", width=350)

        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Classify Flower"):
            with st.spinner("🔍 Bloom scanning in progress..."):
                predicted_class, confidence, top3_classes, top3_confidences = classify_image(temp_path)

            if predicted_class:
                emoji = flower_emojis.get(predicted_class, '🌸')
                st.markdown(f"""
                    <div class='prediction-box'>
                        <h2 style='color:#2e7d32;'>{emoji} Prediction: {predicted_class.capitalize()}</h2>
                        <p style='font-size:18px;'>Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # Store for next tab
                st.session_state["chart_data"] = pd.DataFrame({
                    "Flower": [cls.capitalize() for cls in top3_classes],
                    "Confidence": [conf for conf in top3_confidences]
                })

                try:
                    os.remove(temp_path)
                except:
                    pass

with tab2:
    if "chart_data" in st.session_state:
        st.subheader("Top 3 Predictions")
        st.bar_chart(st.session_state["chart_data"].set_index("Flower"), height=300)
    else:
        st.info("Please upload and classify a flower image in the first tab.")

# Sidebar info
with st.sidebar:
    st.markdown("""
    <div style='background-color:#e8f5e9; padding: 15px; border-radius: 10px; color: #1b5e20;'>
        <h4 style='color:#2e7d32;'>🌼 About the Model</h4>
        <p>This app uses a Convolutional Neural Network (CNN) trained to classify five types of flowers:</p>
        <ul style='margin-left: 20px;'>
            <li>Daisy 🌼</li>
            <li>Dandelion 🌱</li>
            <li>Rose 🌹</li>
            <li>Sunflower 🌻</li>
            <li>Tulip 🌷</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color:#e3f2fd; padding: 15px; border-radius: 10px; margin-top: 20px; color: #0d47a1;'>
        <h4 style='color:#1565c0;'>🔧 How to Use</h4>
        <ol style='margin-left: 20px;'>
            <li>Upload a flower image in JPG, JPEG, or PNG format.</li>
            <li>Click the "Classify Flower" button.</li>
            <li>Check the prediction and top 3 confidence scores.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color:#fff3e0; padding: 15px; border-radius: 10px; margin-top: 20px; color: #e65100;'>
        <h4 style='color:#ef6c00;'>📈 Model Performance</h4>
        <p>✅ Accuracy: ~89% on test data</p>
        <p>🌻 Best Recognized: Sunflower, Tulip</p>
        <p>⚠️ Challenges: Visual similarities between flowers like Daisy and Dandelion</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>Bloom Classifier App • Made with ❤️ using Streamlit and Keras</p>
</div>
""", unsafe_allow_html=True)
