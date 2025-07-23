#!/usr/bin/env python3
"""
Streamlit interface for signature verification
"""
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pytesseract
from PIL import Image
import os

IMAGE_SIZE = (155, 220)  # Fixed size for resizing images

@st.cache_resource
def load_trained_model():
    """Load the trained model (cached for performance)"""
    try:
        model = load_model("signature_verification_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image_for_prediction(image):
    """Preprocess uploaded image for prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if it's a color image
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Resize to model input size
    img_resized = cv2.resize(img_gray, IMAGE_SIZE)
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch and channel dimensions
    img_final = np.expand_dims(img_normalized, axis=0)
    img_final = np.expand_dims(img_final, axis=-1)
    
    return img_final

def extract_text_from_uploaded_image(image):
    """Extract text from uploaded image using OCR"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if it's a color image
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Apply thresholding
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract text
        text = pytesseract.image_to_string(thresh)
        return text.strip()
    except Exception as e:
        st.error(f"Error during OCR: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Signature Verification System",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("✍️ Signature Verification System")
    st.markdown("Upload a signature image to verify if it's genuine or forged using our CNN model.")
    
    # Load the model
    model = load_trained_model()
    if model is None:
        st.error("Failed to load the trained model. Please ensure the model file exists.")
        return
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Signature")
        uploaded_file = st.file_uploader(
            "Choose a signature image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a signature for verification"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Signature", use_column_width=True)
            
            # Preprocess and predict
            processed_image = preprocess_image_for_prediction(image)
            prediction = model.predict(processed_image)[0][0]
            
            # Display prediction
            st.header("Prediction Results")
            if prediction >= 0.5:
                st.success(f"✅ **Genuine Signature** (Confidence: {prediction:.4f})")
            else:
                st.error(f"❌ **Forged Signature** (Confidence: {prediction:.4f})")
            
            # Progress bar for confidence
            st.progress(float(prediction))
            
            # OCR Feature
            st.header("OCR Text Extraction")
            with st.spinner("Extracting text from signature..."):
                extracted_text = extract_text_from_uploaded_image(image)
            
            if extracted_text:
                st.info(f"**Extracted Text:** {extracted_text}")
            else:
                st.warning("No readable text found in the signature.")
    
    with col2:
        st.header("About the Model")
        st.markdown("""
        ### How it works:
        - **CNN Architecture**: Convolutional Neural Network with multiple layers
        - **Training Data**: 720 signature images (360 genuine, 360 forged)
        - **Accuracy**: ~94% on validation set
        - **Input Size**: 155x220 pixels, grayscale
        
        ### Features:
        - ✅ Real-time signature verification
        - ✅ OCR text extraction
        - ✅ Confidence scoring
        - ✅ User-friendly interface
        
        ### Usage Tips:
        - Upload clear, high-contrast signature images
        - Ensure the signature is well-centered
        - Supported formats: PNG, JPG, JPEG
        """)
        
        # Display sample images if they exist
        if os.path.exists("sample_signatures.png"):
            st.header("Sample Training Data")
            st.image("sample_signatures.png", caption="Sample Genuine vs Forged Signatures")
        
        if os.path.exists("confusion_matrix.png"):
            st.header("Model Performance")
            st.image("confusion_matrix.png", caption="Confusion Matrix")
        
        if os.path.exists("training_history_plots.png"):
            st.image("training_history_plots.png", caption="Training History")

if __name__ == "__main__":
    main()

