# 🚀 Quick Start Guide - Skin Disease Classifier

## ⚡ 5-Minute Setup

### 1. Install Python Dependencies (2 min)

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Download Dataset (3 min)

```bash
# Using Kaggle API
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip

# Organize files
mkdir HAM10000_images
mv HAM10000_images_part_1/* HAM10000_images/
mv HAM10000_images_part_2/* HAM10000_images/
```

**Or manually:**
1. Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Download and extract
3. Place images in `./HAM10000_images/`
4. Place metadata in `./HAM10000_metadata.csv`

---

## 🏋️ Training the Model

### Command Line

```bash
python skin_disease_classifier.py
```

**What it does:**
- ✓ Loads 10,000 skin images
- ✓ Trains deep learning model
- ✓ Generates accuracy plots
- ✓ Saves model as `skin_disease_model.h5`

**Training Time:**
- GPU: 30-60 minutes
- CPU: 8-12 hours

---

## 🔮 Making Predictions

### Interactive Mode

```bash
python predict_disease.py
```

**Example:**
```
Enter image path (or 'quit' to exit):
>>> ./test_image.jpg

🎯 PREDICTED DISEASE: Melanoma
   Confidence: 92.45%

📋 Description: Most dangerous type of skin cancer
   Severity: Critical
   Treatment: Surgical excision, immunotherapy
```

### Python Script

```python
from predict_disease import SkinDiseasePredictionEngine

# Load model
engine = SkinDiseasePredictionEngine('skin_disease_model.h5')

# Predict
result = engine.predict('image.jpg')

# Print result
engine.print_result(result)

# Visualize
engine.visualize_prediction('image.jpg')
```

### Batch Predictions

```python
import os
from predict_disease import SkinDiseasePredictionEngine

engine = SkinDiseasePredictionEngine()

for img in os.listdir('./test_images'):
    if img.endswith(('.jpg', '.png')):
        result = engine.predict(f'./test_images/{img}')
        print(f"{img}: {result['predicted_disease_name']}")
```

---

## 📊 Model Output

Each prediction includes:

```python
{
    'predicted_disease_name': 'Melanoma',
    'predicted_disease_code': 'mel',
    'confidence': 0.9245,           # 0.0 to 1.0
    'confidence_percentage': 92.45,  # 0-100%
    'all_predictions': {
        'Melanoma': {'score': 0.9245, 'percentage': 92.45},
        'Nevi': {'score': 0.0321, 'percentage': 3.21},
        ...
    },
    'disease_info': {
        'name': 'Melanoma',
        'description': 'Most dangerous type of skin cancer',
        'severity': 'Critical',
        'treatment': 'Surgical excision, immunotherapy, targeted therapy'
    }
}
```

---

## 🎯 Disease Classes

| Code | Disease | Example |
|------|---------|---------|
| **mel** | Melanoma | Dangerous skin cancer - **HIGH PRIORITY** |
| **bcc** | Basal Cell Carcinoma | Common skin cancer - Medium risk |
| **akiec** | Actinic Keratoses | Precancerous lesions - High risk |
| **bkl** | Benign Keratosis | Non-cancerous moles - Low risk |
| **nv** | Melanocytic Nevi | Common harmless moles - Very Low |
| **df** | Dermatofibroma | Benign skin nodule - Very Low |
| **vasc** | Vascular Lesions | Blood vessel abnormalities - Very Low |

---

## 🐛 Common Issues

### Issue: "Model not found"
```bash
# Train model first
python skin_disease_classifier.py
```

### Issue: "Image not found"
```bash
# Use correct path
result = engine.predict('./images/test.jpg')  # Correct
result = engine.predict('test.jpg')            # Check if file exists
```

### Issue: "Low accuracy"
```python
# In skin_disease_classifier.py
class Config:
    EPOCHS = 100  # Train longer
    BATCH_SIZE = 16  # Smaller batches
    LEARNING_RATE = 5e-5  # Better learning rate
```

### Issue: "Out of memory"
```python
class Config:
    BATCH_SIZE = 8  # Reduce batch size
    IMAGE_SIZE = 128  # Reduce image size
```

---

## 📈 Expected Performance

After training:

| Metric | Target |
|--------|--------|
| Accuracy | 88-92% |
| Precision | 85-91% |
| Recall | 85-90% |
| F1-Score | 85-90% |
| ROC-AUC | 0.95+ |

Per class accuracy:
- Melanoma: 85-90% (most important)
- Nevi: 92-96% (high volume)
- BCC: 80-88%
- Others: 75-85%

---

## 💾 File Structure

```
project/
├── skin_disease_classifier.py      # Training script
├── predict_disease.py               # Prediction script
├── requirements.txt                 # Dependencies
├── SETUP_GUIDE.md                  # Full documentation
├── QUICKSTART.md                   # This file
│
├── HAM10000_images/                # Dataset images (~10,000)
├── HAM10000_metadata.csv           # Dataset metadata
│
├── skin_disease_model.h5           # Trained model (created after training)
├── training_history.png            # Training plots (auto-generated)
└── confusion_matrix.png            # Evaluation matrix (auto-generated)
```

---

## 🔄 Full Workflow

### Step 1: Setup (5 min)
```bash
pip install -r requirements.txt
# Download dataset (see above)
```

### Step 2: Train (1-12 hours depending on hardware)
```bash
python skin_disease_classifier.py
# Outputs: skin_disease_model.h5
```

### Step 3: Predict
```bash
python predict_disease.py
# Enter image path when prompted
```

### Step 4: Integrate (optional)
```python
from predict_disease import SkinDiseasePredictionEngine
engine = SkinDiseasePredictionEngine()
result = engine.predict('my_image.jpg')
```

---

## 🚀 Advanced Usage

### Use Better GPU
```bash
# Install GPU support
pip install tensorflow[and-cuda]==2.13.0

# This will use NVIDIA GPU automatically if available
```

### Use Different Dataset
```python
# In skin_disease_classifier.py, change:
Config.DATASET_PATH = './ISIC_2019'  # Use ISIC instead
```

### Fine-tune Pre-trained Model
```python
# Load trained model
engine = SkinDiseasePredictionEngine()

# Continue training on new data
history = skin_model.train(new_X_train, new_y_train, 
                          new_X_val, new_y_val)
```

### Export to Mobile
```python
# Convert to TFLite for mobile app
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 📚 Learning Resources

### Model Architecture
- **EfficientNetB2**: Efficient and accurate
- **ResNet50**: Strong baseline
- **Vision Transformer**: State-of-the-art

### Dataset
- **Size**: 10,015 images
- **Classes**: 7 skin diseases
- **Verification**: 50%+ histopathology confirmed

### References
- HAM10000 Paper: https://www.nature.com/articles/sdata2018161
- ISIC Challenge: https://challenge.isic-archive.com/

---

## ⚠️ Important Disclaimer

**This model is for RESEARCH & EDUCATION ONLY**

✋ **DO NOT use for:**
- Clinical diagnosis
- Medical decision-making
- Patient treatment planning
- Clinical deployment without validation

✅ **ALWAYS consult** a qualified dermatologist for skin lesion diagnosis

The developers assume NO responsibility for medical decisions based on this model.

---

## 🆘 Need Help?

1. Check `SETUP_GUIDE.md` for detailed instructions
2. Review error messages carefully
3. Verify dataset is correctly placed
4. Check all dependencies are installed
5. Try with CPU first (no GPU issues)

---

**Good luck! 🎯**

For questions or improvements, feel free to contribute!
