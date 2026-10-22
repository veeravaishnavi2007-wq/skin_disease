"""
Skin Disease Prediction Engine
================================
Load a trained model and make predictions on new skin lesion images.
Supports single image, batch processing, and visualization.
"""

import os
import sys
import io
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Fix Windows console encoding issues for emojis and block characters
if sys.platform.startswith('win'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

try:
    import cv2
    import tensorflow as tf
    from tensorflow import keras
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


# ============================================================
# DISEASE INFORMATION DATABASE
# ============================================================

DISEASE_INFO = {
    'mel': {
        'name':        'Melanoma',
        'code':        'mel',
        'severity':    'Critical',
        'risk_level':  '🔴 CRITICAL',
        'description': 'The most dangerous type of skin cancer, originating from melanocytes. '
                       'Early detection is critical as it can spread rapidly to other organs.',
        'symptoms':    'Asymmetric moles, irregular borders, multiple colors, diameter >6mm, evolving lesions.',
        'treatment':   'Surgical excision, immunotherapy (pembrolizumab), targeted therapy (BRAF inhibitors), radiation.',
        'action':      'URGENT: See a dermatologist immediately. Do not delay.',
        'color':       '#FF4444',
        'emoji':       '🔴',
    },
    'bcc': {
        'name':        'Basal Cell Carcinoma',
        'code':        'bcc',
        'severity':    'High',
        'risk_level':  '🟠 HIGH',
        'description': 'The most common skin cancer. Grows slowly and rarely spreads, '
                       'but can cause significant local tissue damage if left untreated.',
        'symptoms':    'Pearly/translucent bump, flat scar-like lesion, bleeding/scabbing sore that heals and returns.',
        'treatment':   'Surgical excision, Mohs surgery, radiation therapy, topical chemotherapy (imiquimod).',
        'action':      'Schedule a dermatology appointment within 2-4 weeks.',
        'color':       '#FF8800',
        'emoji':       '🟠',
    },
    'akiec': {
        'name':        'Actinic Keratoses',
        'code':        'akiec',
        'severity':    'High',
        'risk_level':  '🟠 HIGH',
        'description': 'Precancerous lesions caused by prolonged UV exposure. '
                       'Can progress to squamous cell carcinoma if untreated.',
        'symptoms':    'Rough, scaly patch on sun-damaged skin, itching or burning sensation.',
        'treatment':   'Cryotherapy, topical chemotherapy (5-fluorouracil), photodynamic therapy, laser resurfacing.',
        'action':      'Professional evaluation recommended within 4-6 weeks.',
        'color':       '#FFAA00',
        'emoji':       '🟠',
    },
    'bkl': {
        'name':        'Benign Keratosis',
        'code':        'bkl',
        'severity':    'Low',
        'risk_level':  '🟡 LOW',
        'description': 'Non-cancerous skin growth including seborrheic keratosis, '
                       'solar lentigo, and lichen planus-like keratosis.',
        'symptoms':    'Waxy, stuck-on appearance, rough or smooth texture, varied colors.',
        'treatment':   'Usually no treatment needed. Cryotherapy or curettage if cosmetically bothersome.',
        'action':      'Routine skin check recommended annually.',
        'color':       '#FFDD00',
        'emoji':       '🟡',
    },
    'nv': {
        'name':        'Melanocytic Nevi',
        'code':        'nv',
        'severity':    'Low',
        'risk_level':  '🟢 BENIGN',
        'description': 'Common, benign moles (nevi) formed from melanocyte clusters. '
                       'Most are harmless, but changes in size/color/shape should be monitored.',
        'symptoms':    'Round, well-defined, symmetrical pigmented spot. Uniform color.',
        'treatment':   'No treatment required. Monitor for changes using ABCDE criteria.',
        'action':      'Regular self-examination. Annual skin check recommended.',
        'color':       '#44BB44',
        'emoji':       '🟢',
    },
    'df': {
        'name':        'Dermatofibroma',
        'code':        'df',
        'severity':    'Low',
        'risk_level':  '🟢 BENIGN',
        'description': 'Benign fibrous skin nodule, commonly found on the legs. '
                       'Thought to be a reactive process to minor injury.',
        'symptoms':    'Small, firm bump, dimples inward when pinched, brownish or reddish.',
        'treatment':   'No treatment necessary. Surgical excision if causing discomfort.',
        'action':      'No urgent action required. Mention at next routine checkup.',
        'color':       '#44BBBB',
        'emoji':       '🟢',
    },
    'vasc': {
        'name':        'Vascular Lesions',
        'code':        'vasc',
        'severity':    'Low',
        'risk_level':  '🟢 BENIGN',
        'description': 'Abnormalities of blood vessels in the skin, including hemangiomas, '
                       'angiokeratomas, and pyogenic granulomas.',
        'symptoms':    'Red, purple, or blue skin lesions, may bleed easily.',
        'treatment':   'Laser therapy, sclerotherapy, or surgical removal if indicated.',
        'action':      'No urgent action required. Consult a doctor if lesion grows or bleeds.',
        'color':       '#4488FF',
        'emoji':       '🔵',
    },
}

CLASS_NAMES = sorted(DISEASE_INFO.keys())  # alphabetical


# ============================================================
# PREDICTION ENGINE
# ============================================================

class SkinDiseasePredictionEngine:
    """
    Complete prediction engine for skin disease classification.
    
    Usage:
        engine = SkinDiseasePredictionEngine('skin_disease_model.h5')
        result = engine.predict('my_skin_image.jpg')
        engine.print_result(result)
        engine.visualize_prediction('my_skin_image.jpg', save_path='output.png')
    """
    
    DEFAULT_MODEL_PATH = './skin_disease_model.h5'
    IMAGE_SIZE = 224
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.model = None
        self._load_model()
    
    # ----------------------------------------------------------
    # Model Loading
    # ----------------------------------------------------------
    
    def _load_model(self):
        """Load the Keras model from disk."""
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model not found at: {self.model_path}")
            print("   Run skin_disease_classifier.py to train the model first.")
            print("   The engine will work once a model is available.")
            self.model = None
            return
        
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
    
    # ----------------------------------------------------------
    # Image Preprocessing
    # ----------------------------------------------------------
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image for inference."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)   # (1, H, W, 3)
    
    def preprocess_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess an image from raw bytes (for web API use)."""
        import io
        from PIL import Image
        
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)
    
    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------
    
    def predict(self, image_path: str) -> dict:
        """
        Predict skin disease from a single image.
        
        Args:
            image_path: Path to the image file.
        
        Returns:
            dict with keys:
                - image_path
                - predicted_disease_code
                - predicted_disease_name
                - confidence
                - confidence_percentage
                - all_predictions
                - disease_info
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Train the model first.")
        
        img = self._preprocess_image(image_path)
        probs = self.model.predict(img, verbose=0)[0]
        
        return self._build_result(image_path, probs)
    
    def predict_from_bytes(self, image_bytes: bytes, filename: str = 'uploaded') -> dict:
        """Predict from raw image bytes (used by Flask API)."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        img = self.preprocess_from_bytes(image_bytes)
        probs = self.model.predict(img, verbose=0)[0]
        return self._build_result(filename, probs)
    
    def _build_result(self, image_path: str, probs: np.ndarray) -> dict:
        """Build the structured prediction result dict."""
        pred_idx = int(np.argmax(probs))
        pred_code = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        
        all_predictions = {}
        for i, code in enumerate(CLASS_NAMES):
            score = float(probs[i])
            all_predictions[code] = {
                'code':       code,
                'name':       DISEASE_INFO[code]['name'],
                'score':      score,
                'percentage': round(score * 100, 2),
                'color':      DISEASE_INFO[code]['color'],
                'emoji':      DISEASE_INFO[code]['emoji'],
            }
        
        # Sort by confidence descending
        all_predictions_sorted = dict(
            sorted(all_predictions.items(), key=lambda x: x[1]['score'], reverse=True)
        )
        
        return {
            'image_path':            image_path,
            'predicted_disease_code': pred_code,
            'predicted_disease_name': DISEASE_INFO[pred_code]['name'],
            'confidence':            confidence,
            'confidence_percentage': round(confidence * 100, 2),
            'all_predictions':       all_predictions_sorted,
            'disease_info':          DISEASE_INFO[pred_code],
            'is_critical':           DISEASE_INFO[pred_code]['severity'] in ('Critical', 'High'),
        }
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Run batch predictions on multiple images.
        
        Args:
            image_paths: List of image file paths.
        
        Returns:
            List of prediction dicts (same as predict()).
        """
        from tqdm import tqdm
        results = []
        errors = []
        
        for path in tqdm(image_paths, desc="Predicting"):
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                errors.append({'image_path': path, 'error': str(e)})
        
        if errors:
            print(f"⚠️  {len(errors)} errors during batch prediction")
        
        return results
    
    # ----------------------------------------------------------
    # Output / Display
    # ----------------------------------------------------------
    
    def print_result(self, result: dict):
        """Print a formatted prediction result to the console."""
        sep = "=" * 70
        print(f"\n{sep}")
        print("🏥  SKIN DISEASE PREDICTION RESULT")
        print(sep)
        
        print(f"\n📸  Image: {result['image_path']}")
        print(f"\n🎯  PREDICTED DISEASE: {result['predicted_disease_name']}")
        print(f"    Confidence: {result['confidence_percentage']:.2f}%")
        
        info = result['disease_info']
        print(f"\n📋  Disease Information:")
        print(f"    Risk Level:   {info['risk_level']}")
        print(f"    Description:  {info['description'][:100]}...")
        print(f"    Severity:     {info['severity']}")
        print(f"    Treatment:    {info['treatment']}")
        print(f"\n⚡  Recommended Action: {info['action']}")
        
        print(f"\n📊  Prediction Confidence by Class:")
        for code, pred in result['all_predictions'].items():
            bar_len = int(pred['percentage'] / 100 * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            print(f"    {pred['name']:30s}  {bar}  {pred['percentage']:5.2f}%")
        
        print(f"\n{sep}")
        print("⚠️  DISCLAIMER: For research & education ONLY.")
        print("   Always consult a qualified dermatologist for medical decisions.")
        print(sep)
    
    def visualize_prediction(self, image_path: str, save_path: str = None) -> str:
        """
        Create a rich visualization of the prediction result.
        
        Args:
            image_path: Input image path.
            save_path:  Output PNG path. Auto-generated if None.
        
        Returns:
            Path to the saved visualization PNG.
        """
        result = self.predict(image_path)
        save_path = save_path or image_path.replace('.jpg', '_prediction.png').replace('.jpeg', '_prediction.png').replace('.png', '_prediction.png')
        
        img_orig = cv2.imread(image_path)
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        
        fig = plt.figure(figsize=(16, 8), facecolor='#1a1a2e')
        
        # --- Column 1: Original Image ---
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img_orig)
        ax1.set_title('Input Image', color='white', fontsize=13, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Disease box
        info = result['disease_info']
        color = info['color']
        ax1.text(0.5, -0.1,
                 f"{info['emoji']} {info['name']}",
                 transform=ax1.transAxes,
                 ha='center', va='top',
                 fontsize=14, fontweight='bold',
                 color=color)
        
        # --- Column 2: Confidence Bar Chart ---
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_facecolor('#16213e')
        
        codes = list(result['all_predictions'].keys())
        pcts  = [result['all_predictions'][c]['percentage'] for c in codes]
        names = [result['all_predictions'][c]['name'] for c in codes]
        colors = [result['all_predictions'][c]['color'] for c in codes]
        
        bars = ax2.barh(range(len(codes)), pcts, color=colors, height=0.6, edgecolor='none')
        ax2.set_yticks(range(len(codes)))
        ax2.set_yticklabels(names, color='white', fontsize=10)
        ax2.set_xlabel('Confidence (%)', color='white')
        ax2.set_xlim(0, 100)
        ax2.set_title('Confidence by Class', color='white', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.spines[:].set_color('#333')
        ax2.set_facecolor('#16213e')
        
        for bar, pct in zip(bars, pcts):
            ax2.text(min(pct + 1, 95), bar.get_y() + bar.get_height()/2,
                     f'{pct:.1f}%', va='center', color='white', fontsize=9)
        
        # --- Column 3: Disease Info ---
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_facecolor('#16213e')
        ax3.axis('off')
        
        y = 0.95
        ax3.text(0.05, y, 'Disease Details', fontsize=13, fontweight='bold',
                 color='white', transform=ax3.transAxes, va='top')
        y -= 0.08
        
        details = [
            ('Risk Level',  info['risk_level']),
            ('Severity',    info['severity']),
            ('Confidence',  f"{result['confidence_percentage']:.2f}%"),
            ('',            ''),
            ('Description', info['description'][:80] + '...'),
            ('',            ''),
            ('Treatment',   info['treatment'][:80] + '...'),
            ('',            ''),
            ('Action',      info['action']),
        ]
        
        for label, value in details:
            if label:
                ax3.text(0.05, y, f'{label}:', fontsize=9, fontweight='bold',
                         color='#aaaaaa', transform=ax3.transAxes, va='top')
                ax3.text(0.05, y - 0.05, value, fontsize=9,
                         color='white', transform=ax3.transAxes, va='top',
                         wrap=True)
                y -= 0.13
            else:
                y -= 0.02
        
        # Disclaimer
        ax3.text(0.05, 0.03,
                 '⚠️ For Research & Education Only\nConsult a dermatologist.',
                 fontsize=8, color='#ff9999', transform=ax3.transAxes, va='bottom')
        
        plt.suptitle('🏥 Skin Disease AI Analysis', fontsize=15, fontweight='bold',
                     color='white', y=1.01)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        
        print(f"✅ Visualization saved: {save_path}")
        return save_path


# ============================================================
# INTERACTIVE CLI
# ============================================================

def interactive_cli():
    """Interactive command-line prediction interface."""
    print("\n" + "=" * 60)
    print("🏥  Skin Disease AI Predictor - Interactive Mode")
    print("=" * 60)
    
    model_path = input("\nModel path [skin_disease_model.h5]: ").strip() or 'skin_disease_model.h5'
    
    try:
        engine = SkinDiseasePredictionEngine(model_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\nReady! Enter image paths to analyze (type 'quit' to exit).")
    
    while True:
        print("\n" + "-" * 40)
        path = input("Image path: ").strip()
        
        if path.lower() in ('quit', 'exit', 'q'):
            print("👋 Goodbye!")
            break
        
        if not path:
            continue
        
        try:
            result = engine.predict(path)
            engine.print_result(result)
            
            viz = input("\nGenerate visualization? [y/N]: ").strip().lower()
            if viz == 'y':
                save = input("Save path [auto]: ").strip() or None
                engine.visualize_prediction(path, save)
        
        except FileNotFoundError:
            print(f"❌ Image not found: {path}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == '__main__':
    interactive_cli()
