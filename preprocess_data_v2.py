#!/usr/bin/env python3
"""
Data preprocessing script for signature verification
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMAGE_SIZE = (155, 220)  # Fixed size for resizing images

def preprocess_image(image_path):
    """Preprocess a single image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return None
    img = cv2.resize(img, IMAGE_SIZE)  # Resize to fixed size
    img = img / 255.0  # Normalize pixel values to 0-1
    return img

def load_data_from_structure(dataset_path):
    """Load data from the actual dataset structure"""
    X = []  # To store image data
    y = []  # To store labels (1 for real/original, 0 for forge/forged)
    
    # Look for Dataset_Signature_Final structure
    main_dataset_path = os.path.join(dataset_path, 'Dataset_Signature_Final', 'Dataset')
    
    if not os.path.exists(main_dataset_path):
        # Try alternative path
        main_dataset_path = os.path.join(dataset_path, 'dataset_signature_final', 'Dataset')
    
    if not os.path.exists(main_dataset_path):
        print(f"Dataset path not found: {main_dataset_path}")
        return np.array([]), np.array([])
    
    print(f"Loading data from: {main_dataset_path}")
    
    # Iterate through dataset folders (dataset1, dataset2, etc.)
    for dataset_folder in os.listdir(main_dataset_path):
        dataset_folder_path = os.path.join(main_dataset_path, dataset_folder)
        if not os.path.isdir(dataset_folder_path):
            continue
            
        print(f"Processing {dataset_folder}...")
        
        # Load real signatures
        real_path = os.path.join(dataset_folder_path, 'real')
        if os.path.exists(real_path):
            for img_name in os.listdir(real_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(real_path, img_name)
                    img = preprocess_image(img_path)
                    if img is not None:
                        X.append(img)
                        y.append(1)  # Label as real/original
        
        # Check for real1 folder as well (some datasets have this)
        real1_path = os.path.join(dataset_folder_path, 'real1')
        if os.path.exists(real1_path):
            for img_name in os.listdir(real1_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(real1_path, img_name)
                    img = preprocess_image(img_path)
                    if img is not None:
                        X.append(img)
                        y.append(1)  # Label as real/original
        
        # Load forged signatures
        forge_path = os.path.join(dataset_folder_path, 'forge')
        if os.path.exists(forge_path):
            for img_name in os.listdir(forge_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(forge_path, img_name)
                    img = preprocess_image(img_path)
                    if img is not None:
                        X.append(img)
                        y.append(0)  # Label as forged
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for CNN input (add channel dimension)
    if len(X) > 0:
        X = X.reshape(X.shape[0], IMAGE_SIZE[1], IMAGE_SIZE[0], 1)
    
    return X, y

def explore_data(X, y):
    """Explore the loaded data"""
    print(f"\nData Exploration:")
    print(f"Total images: {len(X)}")
    print(f"Real signatures: {np.sum(y == 1)}")
    print(f"Forged signatures: {np.sum(y == 0)}")
    print(f"Image shape: {X.shape}")
    
    # Plot some sample images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Sample Signatures')
    
    # Show 4 real and 4 forged signatures
    real_indices = np.where(y == 1)[0][:4]
    forged_indices = np.where(y == 0)[0][:4]
    
    for i, idx in enumerate(real_indices):
        axes[0, i].imshow(X[idx].squeeze(), cmap='gray')
        axes[0, i].set_title('Real')
        axes[0, i].axis('off')
    
    for i, idx in enumerate(forged_indices):
        axes[1, i].imshow(X[idx].squeeze(), cmap='gray')
        axes[1, i].set_title('Forged')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_signatures.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Sample signatures saved as 'sample_signatures.png'")

if __name__ == "__main__":
    dataset_base_path = r"E:\signature_verification\dataset"

    X, y = load_data_from_structure(dataset_base_path)
    
    if len(X) == 0:
        print("No data loaded. Please check the dataset path.")
        exit(1)
    
    # Explore the data
    explore_data(X, y)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nData Split:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    # Save preprocessed data for later use
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    print("\nPreprocessed data saved as .npy files.")

