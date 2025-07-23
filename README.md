Signature Verification using CNN & OCR

This is a basic project to verify whether a signature is original or forged using Convolutional Neural Networks (CNN) and OCR with Tesseract.

Dataset:
We use the Handwritten Signatures Dataset from KaggleHub:
https://www.kaggle.com/datasets/divyanshrai/handwritten-signatures

Setup Instructions:

1. Clone the repository
   git clone <your-repo-url>
   cd signature_verification

2. Create and activate virtual environment
   python -m venv venv
   venv\Scripts\activate   (for Windows)

3. Install requirements
   pip install -r requirements.txt

4. Install Tesseract OCR
   - Download from: https://github.com/tesseract-ocr/tesseract
   - Add Tesseract path to your system environment variables
   - Example path: C:\Program Files\Tesseract-OCR\tesseract.exe

5. Run the project
   python main.py

Features:
- CNN model to detect forged vs genuine signature
- OCR via pytesseract to extract embedded text (optional)

Folder Structure:

signature_verification/
│
├── main.py
├── requirements.txt
├── model/
├── data/
│   ├── original/
│   └── forged/
└── README.txt

Author:
Avinash Shinde
