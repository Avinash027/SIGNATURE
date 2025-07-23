
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (155, 220)  # Fixed size for resizing images

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, IMAGE_SIZE)  # Resize to fixed size
    img = img / 255.0  # Normalize pixel values to 0-1
    return img

def load_data(dataset_path):
    X = []  # To store image data
    y = []  # To store labels (1 for original, 0 for forged)

    original_path = os.path.join(dataset_path, 'original_images')
    forged_path = os.path.join(dataset_path, 'forged_images')

    print(f"Loading images from: {original_path}")
    for person_folder in os.listdir(original_path):
        person_original_path = os.path.join(original_path, person_folder)
        if os.path.isdir(person_original_path):
            for img_name in os.listdir(person_original_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_original_path, img_name)
                    X.append(preprocess_image(img_path))
                    y.append(1)  # Label as original

    print(f"Loading images from: {forged_path}")
    for person_folder in os.listdir(forged_path):
        person_forged_path = os.path.join(forged_path, person_folder)
        if os.path.isdir(person_forged_path):
            for img_name in os.listdir(person_forged_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_forged_path, img_name)
                    X.append(preprocess_image(img_path))
                    y.append(0)  # Label as forged

    X = np.array(X)
    y = np.array(y)

    # Reshape for CNN input (add channel dimension)
    X = X.reshape(X.shape[0], IMAGE_SIZE[1], IMAGE_SIZE[0], 1)

    return X, y

if __name__ == "__main__":
    dataset_base_path = "/home/ubuntu/.cache/kagglehub/datasets/divyanshrai/handwritten-signatures/versions/2/sign_data"
    X, y = load_data(dataset_base_path)

    print(f"Total images loaded: {len(X)}")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Save preprocessed data for later use
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    print("Preprocessed data saved as .npy files.")


