#!/usr/bin/env python3
"""
Download the handwritten signatures dataset from Kaggle
"""
import kagglehub
import os

def download_dataset():
    """Download the handwritten signatures dataset"""
    print("Downloading handwritten signatures dataset...")
    path = kagglehub.dataset_download("divyanshrai/handwritten-signatures")
    print(f"Path to dataset files: {path}")
    
    # Create symbolic link for easier access
    if not os.path.exists("dataset"):
        os.symlink(path, "dataset")
        print("Created symbolic link 'dataset' pointing to downloaded data")
    
    return path

if __name__ == "__main__":
    dataset_path = download_dataset()

