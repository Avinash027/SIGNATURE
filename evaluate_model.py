#!/usr/bin/env python3
"""
Model evaluation and visualization script
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

if __name__ == "__main__":
    # Load the trained model
    model = load_model("signature_verification_model.h5")
    
    # Load preprocessed data for evaluation
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")
    
    # Evaluate the model on the validation set
    print("Evaluating model on validation set...")
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    print(f"\nModel Evaluation on Validation Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Forged", "Genuine"],
                yticklabels=["Forged", "Genuine"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")
    
    # Plot training vs validation accuracy and loss
    history = np.load("training_history.npy", allow_pickle=True).item()
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history_plots.png")
    plt.close()
    print("Training history plots saved as training_history_plots.png")


