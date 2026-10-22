# 🏥 Skin Disease Classification Model - Complete Setup Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Training the Model](#training-the-model)
5. [Making Predictions](#making-predictions)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU** (Optional but recommended): NVIDIA GPU with CUDA support
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Disk Space**: ~20GB for dataset + models

### Supported Operating Systems
- ✅ Windows 10/11
- ✅ macOS
- ✅ Linux (Ubuntu 18.04+)

---

## Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# For Windows
python -m venv skin_disease_env
skin_disease_env\Scripts\activate

# For macOS/Linux
python3 -m venv skin_disease_env
source skin_disease_env/bin/activate
```

### Step 2: Install Required Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install TensorFlow and dependencies
pip install tensorflow>=2.13.0
pip install keras>=2.13.0

# Install data processing libraries
pip install numpy pandas scikit-learn opencv-python pillow

# Install visualization libraries
pip install matplotlib seaborn

# Install additional utilities
pip install tqdm kaggle
```

### Step 3: Verify Installation

```python
python -c "import tensorflow as tf; print(f'TensorFlow Version: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV Version: {cv2.__version__}')"
```

**Expected Output:**
```
TensorFlow Version: 2.13.0 (or higher)
OpenCV Version: 4.x.x (or higher)
```

---

## Dataset Setup

### Option 1: Download from Kaggle (Recommended)

#### Step 1a: Get Kaggle API Credentials
1. Visit: https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`

#### Step 1b: Setup Kaggle API
```bash
# Create kaggle directory
# Windows
mkdir %USERPROFILE%\.kaggle
# Copy kaggle.json to this directory

# macOS/Linux
mkdir -p ~/.kaggle
# Copy kaggle.json to this directory
chmod 600 ~/.kaggle/kaggle.json
```

#### Step 1c: Download HAM10000 Dataset
```bash
# Download the dataset
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Extract the dataset
unzip skin-cancer-mnist-ham10000.zip

# You should now have:
# - HAM10000_images_part_1/
# - HAM10000_images_part_2/
# - HAM10000_metadata.csv
```

#### Step 1d: Organize Files
```bash
# Create images folder and merge both parts
mkdir HAM10000_images

# Move all images to single folder
mv HAM10000_images_part_1/* HAM10000_images/
mv HAM10000_images_part_2/* HAM10000_images/

# Verify you have ~10,000 images
ls HAM10000_images | wc -l
```

### Option 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click "Download" button
3. Extract downloaded file
4. Organize as shown above

### Option 3: Use ISIC Challenge Dataset

```bash
# Alternative: Download from ISIC Archive
# https://challenge.isic-archive.com/data/

# Large dataset with more images
# ~25,000+ images for ISIC 2019
```

### Verify Dataset Structure
```
your_project_folder/
├── HAM10000_images/           # All image files
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ... (~10,000 images)
├── HAM10000_metadata.csv       # Metadata file
├── skin_disease_classifier.py
├── predict_disease.py
└── SETUP_GUIDE.md
```

**Dataset Statistics:**
```
Total Images: 10,015
Classes: 7
Distribution:
- Melanocytic Nevi (nv): 6,705
- Melanoma (mel): 1,113
- Benign Keratosis (bkl): 1,099
- Basal Cell Carcinoma (bcc): 514
- Actinic Keratoses (akiec): 327
- Vascular Lesions (vasc): 143
- Dermatofibroma (df): 115
```

---

## Training the Model

### Step 1: Basic Training

```bash
# Run training script
python skin_disease_classifier.py
```

### Step 2: Monitor Training

The script will:
1. ✓ Load and verify dataset
2. ✓ Preprocess all images
3. ✓ Split into train/val/test (70/15/15)
4. ✓ Build EfficientNetB2 model
5. ✓ Train for up to 50 epochs
6. ✓ Evaluate on test set
7. ✓ Generate visualizations
8. ✓ Save model as `skin_disease_model.h5`

### Step 3: Expected Training Time

| Hardware | Time (hours) |
|----------|------------|
| CPU Only | 8-12+ hours |
| GPU (NVIDIA GTX 1080) | 30-60 minutes |
| GPU (NVIDIA V100) | 15-20 minutes |
| TPU | 10-15 minutes |

### Step 4: Monitor Training Progress

**Console Output Should Show:**
```
Epoch 1/50
200/200 [==============================] - 45s 224ms/step
loss: 0.5432 - accuracy: 0.7823 - val_loss: 0.4123 - val_accuracy: 0.8234

Epoch 2/50
200/200 [==============================] - 42s 209ms/step
loss: 0.3421 - accuracy: 0.8654 - val_loss: 0.3012 - val_accuracy: 0.8876
...
```

### Step 5: Performance Metrics

The model should achieve:
- **Training Accuracy**: 92-96%
- **Validation Accuracy**: 88-94%
- **Test Accuracy**: 85-92%
- **ROC-AUC**: 0.95+

### Step 6: Advanced Training (Optional)

#### Use GPU (NVIDIA)
```bash
# Install CUDA support
pip install tensorflow[and-cuda]==2.13.0

# Or use GPU-optimized TensorFlow
pip install tensorflow-gpu
```

#### Fine-tune Hyperparameters
Edit `Config` class in `skin_disease_classifier.py`:
```python
class Config:
    BATCH_SIZE = 32          # Increase for faster training
    EPOCHS = 50              # More epochs = better accuracy (diminishing returns)
    LEARNING_RATE = 1e-4     # Smaller = more stable, slower
    IMAGE_SIZE = 224         # Larger = more accurate, slower
```

#### Use Different Architecture
```python
# In skin_disease_classifier.py, choose one:
skin_model.build_model_v1_efficient()  # Fastest, good accuracy
skin_model.build_model_v2_resnet()     # Slower, slightly better accuracy
```

---

## Making Predictions

### Option 1: Interactive Prediction

```bash
# Run prediction script
python predict_disease.py
```

**Usage:**
```
Enter image path (or 'quit' to exit):
>>> /path/to/skin_image.jpg

[PREDICTION RESULTS]
```

### Option 2: Programmatic Prediction

```python
from predict_disease import SkinDiseasePredictionEngine

# Initialize engine
engine = SkinDiseasePredictionEngine(model_path='skin_disease_model.h5')

# Make single prediction
result = engine.predict('path_to_image.jpg')

# Print formatted result
engine.print_result(result)

# Visualize prediction
engine.visualize_prediction('path_to_image.jpg', save_path='prediction.png')

# Access prediction data
disease_name = result['predicted_disease_name']
confidence = result['confidence_percentage']
treatment = result['disease_info']['treatment']

print(f"Disease: {disease_name}")
print(f"Confidence: {confidence:.2f}%")
print(f"Treatment: {treatment}")
```

### Option 3: Batch Prediction

```python
import os
from predict_disease import SkinDiseasePredictionEngine

engine = SkinDiseasePredictionEngine()

# Predict on all images in folder
image_folder = './test_images'
results = []

for img_file in os.listdir(image_folder):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, img_file)
        result = engine.predict(img_path)
        results.append(result)
        
        # Print summary
        print(f"\n{img_file}: {result['predicted_disease_name']} ({result['confidence_percentage']:.2f}%)")

# Save results to CSV
import csv
with open('predictions.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image', 'disease', 'confidence'])
    writer.writeheader()
    for r in results:
        writer.writerow({
            'image': r['image_path'],
            'disease': r['predicted_disease_name'],
            'confidence': f"{r['confidence_percentage']:.2f}%"
        })
```

### Prediction Output Format

```
======================================================================
🏥 SKIN DISEASE PREDICTION RESULT
======================================================================

📸 Image: /path/to/image.jpg

🎯 PREDICTED DISEASE: Melanoma
   Confidence: 92.45%

📋 Disease Information:
   Description: Most dangerous type of skin cancer
   Severity: Critical
   Treatment Options: Surgical excision, immunotherapy, targeted therapy

📊 Prediction Confidence by Class:
   Melanoma                      ████████████████████████ 92.45%
   Melanocytic Nevi             ████                      3.21%
   Basal Cell Carcinoma         ██                        1.87%
   Benign Keratosis             █                         1.12%
   Actinic Keratoses            █                         0.89%
   Dermatofibroma               █                         0.32%
   Vascular Lesions             ░                         0.14%

======================================================================
⚠️  DISCLAIMER:
   This prediction is for reference only and should NOT be used
   as a substitute for professional medical diagnosis.
   Please consult a dermatologist for accurate diagnosis.
======================================================================
```

---

## Troubleshooting

### ❌ Problem: "Module not found: tensorflow"

**Solution:**
```bash
pip install --upgrade tensorflow
```

### ❌ Problem: "CUDA out of memory"

**Solutions:**
1. Reduce batch size: `BATCH_SIZE = 16` (in Config)
2. Reduce image size: `IMAGE_SIZE = 128` (in Config)
3. Use CPU only: Uninstall tensorflow-gpu
4. Close other applications consuming GPU memory

### ❌ Problem: "HAM10000_metadata.csv not found"

**Solution:**
```bash
# Verify file exists
ls -la HAM10000_metadata.csv

# If missing, download again from Kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
```

### ❌ Problem: "Image file not found during training"

**Solution:**
```bash
# Verify image files exist
ls HAM10000_images | head -10

# Should show image files like:
# ISIC_0024306.jpg
# ISIC_0024307.jpg
```

### ❌ Problem: "Model accuracy is too low (<80%)"

**Solutions:**
1. **Check data quality**: Ensure images are correctly placed
2. **Train longer**: Increase EPOCHS to 100
3. **Increase data**: Use ISIC 2019 dataset (more images)
4. **Better learning rate**: Try 1e-5 or 5e-5
5. **Add data augmentation**: Already enabled, check if working

### ❌ Problem: "Prediction returns completely different disease"

**Solutions:**
1. Ensure image is actual skin lesion
2. Check image quality (clear, focused, good lighting)
3. Try different images to verify model accuracy
4. Retrain with better hyperparameters

### ❌ Problem: "Out of RAM during training"

**Solutions:**
```bash
# Reduce batch size
BATCH_SIZE = 8  # Even smaller

# Reduce number of images loaded at once
# Edit skin_disease_classifier.py
# Enable data generator mode (streaming)

# Or use Google Colab with free GPU
# https://colab.research.google.com/
```

---

## Performance Optimization

### GPU Acceleration (NVIDIA)

```bash
# Install CUDA 11.8 for TensorFlow 2.13
# https://developer.nvidia.com/cuda-11-8-0-download-archive

# Install cuDNN 8.6
# https://developer.nvidia.com/cudnn

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]==2.13.0
```

### Mixed Precision Training (Faster)

```python
# Add to skin_disease_classifier.py
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# This can speed up training by 2-3x on modern GPUs
```

### Data Augmentation

Already enabled in code:
- Random rotation (±20°)
- Horizontal/vertical flip
- Random zoom
- Brightness adjustment
- Shift augmentation

---

## Model Deployment

### Save Model for Production

```python
# Already done by skin_disease_classifier.py
# Model saved as: skin_disease_model.h5

# Convert to TF Lite for mobile
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('skin_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Create Web API

```python
# Using Flask
from flask import Flask, request, jsonify
from predict_disease import SkinDiseasePredictionEngine

app = Flask(__name__)
engine = SkinDiseasePredictionEngine()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file.save('temp_image.jpg')
    
    result = engine.predict('temp_image.jpg')
    
    return jsonify({
        'disease': result['predicted_disease_name'],
        'confidence': result['confidence_percentage']
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

---

## Additional Resources

### Documentation
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- OpenCV: https://docs.opencv.org/

### Datasets
- HAM10000: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- ISIC Archive: https://challenge.isic-archive.com/data/
- Melanoma Dataset: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

### Research Papers
- HAM10000 Paper: https://www.nature.com/articles/sdata2018161
- EfficientNet: https://arxiv.org/abs/1905.11946
- Melanoma Detection Review: https://arxiv.org/abs/2411.12846

---

## Support & Questions

If you encounter issues:
1. Check the Troubleshooting section above
2. Review error messages carefully
3. Check dataset is correctly placed
4. Verify all dependencies are installed
5. Try with smaller dataset first (for testing)

---

## License & Disclaimer

⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This model is for **educational and research purposes only**. 

- Do NOT use this model for clinical decision-making
- Do NOT rely on this model as a substitute for professional dermatological diagnosis
- Always consult a qualified dermatologist for skin lesion diagnosis
- Model predictions should only be used to support, not replace, expert medical opinion

The developers are NOT responsible for any medical decisions made based on this model's predictions.

---

**Happy Training! 🚀**
