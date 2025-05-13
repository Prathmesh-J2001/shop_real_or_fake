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
st.write("Upload a front-view image of a shop to check if it's real or fake.")

# Upload image
uploaded_file = st.file_uploader("Choose a shop image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = class_names[1] if prediction > 0.5 else class_names[0]

    # Output
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.progress(int(prediction * 100 if prediction > 0.5 else (1 - prediction) * 100))
