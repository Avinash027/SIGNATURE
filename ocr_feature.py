#!/usr/bin/env python3
"""
OCR feature for extracting text from signature images using pytesseract.
"""
import numpy as np
import cv2
import pytesseract
import os

def extract_text_from_image(image_path):
    """Extract text from an image using pytesseract."""
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a binary image (optional, but often helps OCR)
        # You might need to adjust the threshold value based on your images
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(thresh)
        return text.strip()
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

if __name__ == "__main__":
    # Create a dummy image for testing OCR
    os.makedirs("ocr_test_images", exist_ok=True)
    dummy_ocr_image_path = "ocr_test_images/dummy_text.png"
    
    # Create a simple image with text using OpenCV (for demonstration)
    # In a real scenario, you would use an actual signature image with legible text
    dummy_img_with_text = np.zeros((100, 300, 3), dtype=np.uint8) + 255 # White background
    cv2.putText(dummy_img_with_text, "Signature", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.imwrite(dummy_ocr_image_path, dummy_img_with_text)
    
    print(f"\n--- Testing OCR with dummy image: {dummy_ocr_image_path} ---")
    extracted_text = extract_text_from_image(dummy_ocr_image_path)
    if extracted_text:
        print(f"Extracted Text: \"{extracted_text}\"")
    else:
        print("No text extracted.")

    # Clean up dummy image
    os.remove(dummy_ocr_image_path)
    os.rmdir("ocr_test_images")
    print("Cleaned up dummy OCR test image and folder.")


