#!/usr/bin/env python3
"""
CNN model training script for signature verification
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

IMAGE_SIZE = (155, 220)  # Fixed size for resizing images

def build_cnn_model(input_shape):
    """Build and compile CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')

    print(f"Loaded X_train shape: {X_train.shape}")
    print(f"Loaded X_val shape: {X_val.shape}")

    input_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], 1)  # Height, Width, Channels
    model = build_cnn_model(input_shape)
    model.summary()

    # Define Early Stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=20,  # Train for ~10-20 epochs
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # Save the trained model
    model.save('signature_verification_model.h5')
    print("Model trained and saved as signature_verification_model.h5")

    # Save training history for plotting
    np.save('training_history.npy', history.history)
    print("Training history saved as training_history.npy")

