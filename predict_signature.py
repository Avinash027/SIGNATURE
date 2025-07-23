#!/usr/bin/env python3
"""
Script for predicting whether a user-provided signature is genuine or forged.
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

IMAGE_SIZE = (155, 220)  # Fixed size for resizing images

def preprocess_single_image(image_path):
    """Preprocess a single image for prediction."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    img = cv2.resize(img, IMAGE_SIZE)  # Resize to fixed size
    img = img / 255.0  # Normalize pixel values to 0-1
    # Reshape for CNN input (add batch and channel dimensions)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1) # Add channel dimension
    return img

def predict_signature(image_path, model_path="signature_verification_model.h5"):
    """Predict if a signature is genuine or forged."""
    # Load the trained model
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Preprocess the input image
    processed_img = preprocess_single_image(image_path)
    if processed_img is None:
        return None

    # Make prediction
    prediction = model.predict(processed_img)[0][0]

    if prediction >= 0.5:
        return "Genuine", prediction
    else:
        return "Forged", prediction

if __name__ == "__main__":
    # Example usage (replace with actual image paths for testing)
    # To test, you would need to place an image in original/user_signature.jpg or forged/user_signature.jpg
    # For demonstration, let's assume a sample image exists.
    
    # Create dummy folders and a dummy image for testing purposes
    os.makedirs("test_signatures/original", exist_ok=True)
    os.makedirs("test_signatures/forged", exist_ok=True)
    
    # Create a dummy image (a black image)
    dummy_image = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)
    cv2.imwrite("test_signatures/original/dummy_genuine.png", dummy_image)
    cv2.imwrite("test_signatures/forged/dummy_forged.png", dummy_image)
    
    print("\n--- Testing with dummy genuine signature ---")
    result, confidence = predict_signature("test_signatures/original/dummy_genuine.png")
    if result:
        print(f"Prediction: {result} (Confidence: {confidence:.4f})")

    print("\n--- Testing with dummy forged signature ---")
    result, confidence = predict_signature("test_signatures/forged/dummy_forged.png")
    if result:
        print(f"Prediction: {result} (Confidence: {confidence:.4f})")

    print("\nNote: For real-world testing, replace dummy images with actual signature images.")


