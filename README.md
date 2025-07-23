# Signature Verification using CNN (Handwritten Signatures Dataset)

A comprehensive machine learning project that implements a Convolutional Neural Network (CNN) for signature verification, distinguishing between genuine and forged signatures using deep learning techniques.

## 🎯 Project Overview

This project builds a robust signature verification system using a CNN model trained on the Kaggle handwritten signatures dataset. The system can classify whether a given signature is genuine or forged with high accuracy, providing a practical solution for document authentication and fraud detection.

### Key Features

- **CNN-based Classification**: Deep learning model with 94% accuracy on validation data
- **Real-time Prediction**: Fast signature verification for uploaded images
- **OCR Integration**: Text extraction from signature images using Tesseract
- **Interactive Web Interface**: User-friendly Streamlit application
- **Comprehensive Evaluation**: Detailed model performance metrics and visualizations
- **Easy Deployment**: Ready-to-use scripts and documentation

## 📊 Dataset Information

The project uses the `divyanshrai/handwritten-signatures` dataset from Kaggle, which contains:

- **Total Images**: 720 signature samples
- **Genuine Signatures**: 360 samples (labeled as 1)
- **Forged Signatures**: 360 samples (labeled as 0)
- **Image Format**: PNG/JPG files
- **Resolution**: Variable (resized to 155x220 pixels for training)

## 🏗️ Architecture

### CNN Model Structure

```
Input Layer: (220, 155, 1) - Grayscale images
├── Conv2D(32, 3x3) + ReLU + MaxPooling(2x2)
├── Conv2D(64, 3x3) + ReLU + MaxPooling(2x2)
├── Conv2D(128, 3x3) + ReLU + MaxPooling(2x2)
├── Flatten
├── Dense(128) + ReLU + Dropout(0.5)
└── Dense(1) + Sigmoid (Binary Classification)
```

**Total Parameters**: 7,056,129 (26.92 MB)

### Performance Metrics

- **Validation Accuracy**: 94.44%
- **Precision**: 97.06%
- **Recall**: 91.67%
- **Training Time**: ~20 epochs with early stopping

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+ required
pip install tensorflow opencv-python matplotlib kagglehub pytesseract streamlit scikit-learn seaborn
```

### Installation

1. **Clone and setup the project**:
```bash
mkdir signature_verification
cd signature_verification
```

2. **Download the dataset**:
```bash
python3 download_dataset.py
```

3. **Preprocess the data**:
```bash
python3 preprocess_data_v2.py
```

4. **Train the model**:
```bash
python3 train_model.py
```

5. **Evaluate the model**:
```bash
python3 evaluate_model.py
```

6. **Run the Streamlit interface**:
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
signature_verification/
├── download_dataset.py          # Dataset download script
├── preprocess_data_v2.py        # Data preprocessing and exploration
├── train_model.py               # CNN model training
├── evaluate_model.py            # Model evaluation and visualization
├── predict_signature.py         # Single image prediction
├── ocr_feature.py              # OCR text extraction
├── streamlit_app.py            # Web interface
├── signature_verification_model.h5  # Trained model
├── X_train.npy, X_val.npy      # Preprocessed data
├── y_train.npy, y_val.npy      # Labels
├── training_history.npy        # Training metrics
├── sample_signatures.png       # Sample data visualization
├── confusion_matrix.png        # Model performance
├── training_history_plots.png  # Training curves
└── README.md                   # This file
```

## 🔧 Usage Examples

### Command Line Prediction

```python
from predict_signature import predict_signature

# Predict a single signature
result, confidence = predict_signature("path/to/signature.jpg")
print(f"Prediction: {result} (Confidence: {confidence:.4f})")
```

### OCR Text Extraction

```python
from ocr_feature import extract_text_from_image

# Extract text from signature
text = extract_text_from_image("path/to/signature.jpg")
print(f"Extracted text: {text}")
```

### Web Interface

Launch the Streamlit app for an interactive experience:

```bash
streamlit run streamlit_app.py
```

Features include:
- Drag-and-drop file upload
- Real-time prediction results
- OCR text extraction
- Model performance visualization
- Confidence scoring with progress bars

## 📈 Model Performance

### Training Results

The model achieved excellent performance on the validation set:

| Metric | Value |
|--------|-------|
| Accuracy | 94.44% |
| Precision | 97.06% |
| Recall | 91.67% |
| F1-Score | 94.29% |

### Confusion Matrix

```
                Predicted
Actual    Forged  Genuine
Forged      70      2
Genuine      6     66
```

### Training Curves

The model showed stable convergence with minimal overfitting:
- Training accuracy reached 99.2% by epoch 20
- Validation accuracy peaked at 97.2%
- Early stopping prevented overfitting

## 🔍 Technical Details

### Data Preprocessing

1. **Image Loading**: Load images from dataset structure
2. **Grayscale Conversion**: Convert to single channel
3. **Resizing**: Standardize to 155x220 pixels
4. **Normalization**: Scale pixel values to [0, 1]
5. **Train-Test Split**: 80-20 split with stratification

### Model Training

- **Optimizer**: Adam with default learning rate
- **Loss Function**: Binary crossentropy
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)
- **Callbacks**: Early stopping with patience=5

### OCR Implementation

- **Engine**: Tesseract OCR
- **Preprocessing**: Grayscale conversion and thresholding
- **Text Extraction**: OTSU thresholding for optimal results

## 🛠️ Customization

### Adding New Signatures

1. Create folders for your signatures:
```bash
mkdir -p test_signatures/original
mkdir -p test_signatures/forged
```

2. Add your signature images to the appropriate folders

3. Test with the prediction script:
```bash
python3 predict_signature.py
```

### Model Retraining

To retrain with additional data:

1. Add new images to the dataset structure
2. Run preprocessing: `python3 preprocess_data_v2.py`
3. Retrain the model: `python3 train_model.py`
4. Evaluate performance: `python3 evaluate_model.py`

### Hyperparameter Tuning

Modify `train_model.py` to experiment with:
- Different CNN architectures
- Learning rates and optimizers
- Batch sizes and epochs
- Dropout rates and regularization

## 🚀 Deployment Options

### Local Deployment

```bash
# Run Streamlit locally
streamlit run streamlit_app.py --server.port 8501
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment

The application can be deployed on:
- Streamlit Cloud
- Heroku
- AWS EC2
- Google Cloud Platform
- Azure Container Instances

## 🔬 Future Enhancements

### Planned Features

1. **Multi-class Classification**: Support for multiple signature types
2. **Real-time Video Processing**: Live signature verification
3. **Mobile App**: React Native or Flutter implementation
4. **API Endpoints**: RESTful API for integration
5. **Database Integration**: Store and manage signature records
6. **Advanced OCR**: Handwriting recognition improvements
7. **Ensemble Methods**: Combine multiple models for better accuracy

### Research Directions

1. **Siamese Networks**: For one-shot learning approaches
2. **Attention Mechanisms**: Focus on signature-specific features
3. **Data Augmentation**: Synthetic signature generation
4. **Transfer Learning**: Pre-trained models for better performance
5. **Explainable AI**: Visualization of model decision-making

## 📚 Dependencies

### Core Libraries

```
tensorflow>=2.19.0
opencv-python>=4.12.0
numpy>=2.1.3
matplotlib>=3.8.2
scikit-learn>=1.7.1
streamlit>=1.47.0
pytesseract>=0.3.13
seaborn>=0.13.2
kagglehub>=0.3.12
```

### System Requirements

- **Python**: 3.11+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dataset and models
- **OS**: Linux, macOS, or Windows
- **Tesseract**: For OCR functionality

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** with proper documentation
4. **Add tests** for new functionality
5. **Submit a pull request** with detailed description

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd signature_verification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Thanks to Divyansh Rai for the handwritten signatures dataset on Kaggle
- **Libraries**: TensorFlow, OpenCV, Streamlit, and the Python community
- **Inspiration**: Research papers on signature verification and CNN applications

## 📞 Support

For questions, issues, or contributions:

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: Contact the maintainers

## 📊 Changelog

### Version 1.0.0 (Current)
- Initial release with CNN model
- Streamlit web interface
- OCR text extraction
- Comprehensive documentation
- Model evaluation tools

---

**Built with ❤️ by Manus AI**

*Last updated: July 2025*

#   S I G N A T U R E  
 