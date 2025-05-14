import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("shop_real_vs_fake_model.h5")

# Class labels
class_names = ['Fake', 'Real']

# Title
st.title("🛍️ Shop Real vs Fake Classifier")
st.write("Upload front-view images of shops to check if they are real or fake.")

# Upload multiple images
uploaded_files = st.file_uploader("Choose one or more shop images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write("---")  # Separator between images

        # Display image
        img = Image.open(uploaded_file)
        st.image(img, caption=uploaded_file.name, use_column_width=True)

        # Preprocess
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        label = class_names[1] if prediction > 0.5 else class_names[0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Output
        st.markdown(f"### 🧠 Prediction: **{label}**")
        st.progress(int(confidence * 100))
