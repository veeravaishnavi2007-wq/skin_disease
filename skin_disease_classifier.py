import os
import sys
import io
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Fix Windows console encoding issues for emojis and block characters
if sys.platform.startswith('win'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, callbacks
    from tensorflow.keras.applications import EfficientNetB2, ResNet50
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, confusion_matrix
    import cv2
    print(f"✅ TensorFlow {tf.__version__} loaded successfully")
    print(f"✅ GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Central configuration for the skin disease classifier."""
    
    # Paths
    DATASET_PATH = './HAM10000_images'
    METADATA_PATH = './HAM10000_metadata.csv'
    MODEL_SAVE_PATH = './skin_disease_model.h5'
    
    # Model parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    DROPOUT_RATE = 0.3
    
    # Training
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    PATIENCE = 10
    MIN_LR = 1e-7
    LR_FACTOR = 0.5
    LR_PATIENCE = 5
    
    # Disease classes mapping
    DISEASE_CLASSES = {
        'akiec': 'Actinic Keratoses',
        'bcc':   'Basal Cell Carcinoma',
        'bkl':   'Benign Keratosis',
        'df':    'Dermatofibroma',
        'mel':   'Melanoma',
        'nv':    'Melanocytic Nevi',
        'vasc':  'Vascular Lesions',
    }
    
    CLASS_NAMES = sorted(DISEASE_CLASSES.keys())  # alphabetical
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Colors for visualization
    COLORS = {
        'mel':   '#FF4444',   # Red  - Critical
        'bcc':   '#FF8800',   # Orange - High
        'akiec': '#FFAA00',   # Amber - High
        'bkl':   '#FFDD00',   # Yellow - Medium
        'nv':    '#44BB44',   # Green - Low
        'df':    '#44BBBB',   # Teal - Low
        'vasc':  '#4488FF',   # Blue - Low
    }
    
    RANDOM_SEED = 42


# ============================================================
# DATA LOADER
# ============================================================

class SkinDataLoader:
    """Loads and preprocesses the HAM10000 dataset using memory-efficient tf.data pipelines."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.metadata = None
        self.class_weights = None
        
    def load_metadata(self):
        """Load and validate the metadata CSV."""
        path = self.config.METADATA_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Metadata file not found: {path}\n"
                "Please download HAM10000_metadata.csv from Kaggle."
            )
        
        self.metadata = pd.read_csv(path)
        print(f"✅ Metadata loaded: {len(self.metadata)} entries")
        
        # Add full image path
        self.metadata['filepath'] = self.metadata['image_id'].apply(self.get_image_path)
        # Drop entries where image was not found
        missing_count = self.metadata['filepath'].isna().sum()
        if missing_count > 0:
            print(f"⚠️  {missing_count} images listed in metadata but not found in {self.config.DATASET_PATH}. Dropping.")
            self.metadata = self.metadata.dropna(subset=['filepath']).reset_index(drop=True)
            
        # Map labels to indices
        self.metadata['label_idx'] = self.metadata['dx'].apply(lambda x: self.config.CLASS_NAMES.index(x))
        
        # Show class distribution
        print("\n📊 Class Distribution:")
        dist = self.metadata['dx'].value_counts()
        for cls, count in dist.items():
            name = self.config.DISEASE_CLASSES.get(cls, cls)
            pct = count / len(self.metadata) * 100
            print(f"   {cls:8s} ({name:30s}): {count:5d} ({pct:.1f}%)")
            
        # Compute class weights
        y_labels = self.metadata['label_idx'].values
        class_w = compute_class_weight(
            'balanced',
            classes=np.unique(y_labels),
            y=y_labels
        )
        self.class_weights = dict(enumerate(class_w))
        print(f"✅ Computed class weights: {self.class_weights}")
        
        return self.metadata
        
    def get_image_path(self, image_id):
        """Get full path for an image ID."""
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(self.config.DATASET_PATH, image_id + ext)
            if os.path.exists(path):
                return path
        return None

    def split_and_build_datasets(self):
        """Split the metadata and construct tf.data.Dataset objects."""
        if self.metadata is None:
            self.load_metadata()
            
        # Stratified train/val/test split of dataframes
        train_df, temp_df = train_test_split(
            self.metadata,
            test_size=self.config.VALIDATION_SPLIT + self.config.TEST_SPLIT,
            random_state=self.config.RANDOM_SEED,
            stratify=self.metadata['label_idx']
        )
        
        rel_test = self.config.TEST_SPLIT / (self.config.VALIDATION_SPLIT + self.config.TEST_SPLIT)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=rel_test,
            random_state=self.config.RANDOM_SEED,
            stratify=temp_df['label_idx']
        )
        
        print(f"\n📊 Stratified Data Splits:")
        print(f"   Train:      {len(train_df)} rows")
        print(f"   Validation: {len(val_df)} rows")
        print(f"   Test:       {len(test_df)} rows")
        
        # Local standalone function to map image loading (avoids pickling class method self reference)
        image_size = self.config.IMAGE_SIZE
        def parse_image_fn(filepath, label):
            img = tf.io.read_file(filepath)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [image_size, image_size])
            img = tf.cast(img, tf.float32) / 255.0
            return img, label
            
        # Build tf.data Datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_df['filepath'].values, train_df['label_idx'].values))
        train_ds = train_ds.shuffle(buffer_size=2048, seed=self.config.RANDOM_SEED)
        train_ds = train_ds.map(parse_image_fn, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(self.config.BATCH_SIZE)
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((val_df['filepath'].values, val_df['label_idx'].values))
        val_ds = val_ds.map(parse_image_fn, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.config.BATCH_SIZE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((test_df['filepath'].values, test_df['label_idx'].values))
        test_ds = test_ds.map(parse_image_fn, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.config.BATCH_SIZE)
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return train_ds, val_ds, test_ds, test_df


# ============================================================
# MODEL BUILDER
# ============================================================

class SkinDiseaseModel:
    """Builds and manages the skin disease classification model."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = None
        self.history = None
    
    def build_model_v1_efficient(self):
        """
        Build EfficientNetB2-based classifier (recommended).
        Fast inference, compact size, high accuracy.
        """
        print("\n🏗️  Building EfficientNetB2 model...")
        
        base = EfficientNetB2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3)
        )
        
        # Freeze base layers initially
        base.trainable = False
        
        inputs = keras.Input(shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3))
        
        # Data augmentation (applied during training)
        x = layers.RandomFlip("horizontal_and_vertical")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomBrightness(0.2)(x)
        x = layers.RandomContrast(0.1)(x)
        
        # Base model
        x = base(x, training=False)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        self.model = Model(inputs, outputs, name='SkinDiseaseClassifier_EfficientNet')
        
        self._compile_model()
        self.model.summary()
        
        print(f"✅ Model built: {self.model.count_params():,} parameters")
        return self.model
    
    def build_model_v2_resnet(self):
        """
        Build ResNet50-based classifier (alternative).
        Slightly higher accuracy, larger size.
        """
        print("\n🏗️  Building ResNet50 model...")
        
        base = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3)
        )
        base.trainable = False
        
        inputs = keras.Input(shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3))
        
        x = layers.RandomFlip("horizontal_and_vertical")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.1)(x)
        
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        self.model = Model(inputs, outputs, name='SkinDiseaseClassifier_ResNet')
        self._compile_model()
        self.model.summary()
        
        print(f"✅ Model built: {self.model.count_params():,} parameters")
        return self.model
    
    def _compile_model(self):
        """Compile model with Adam optimizer and sparse categorical crossentropy."""
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )
    
    def _get_callbacks(self):
        """Build training callbacks."""
        return [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.LR_FACTOR,
                patience=self.config.LR_PATIENCE,
                min_lr=self.config.MIN_LR,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=self.config.MODEL_SAVE_PATH.replace('.h5', '_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=f'./logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=0
            ),
        ]
    
    def train(self, train_ds, val_ds, class_weights=None):
        """Train the model with callbacks and class balancing."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model_v1_efficient() first.")
        
        print(f"\n🏋️  Starting training...")
        print(f"   Epochs: {self.config.EPOCHS}")
        print(f"   Batch size: {self.config.BATCH_SIZE}")
        print(f"   Learning rate: {self.config.LEARNING_RATE}")
        
        self.history = self.model.fit(
            train_ds,
            epochs=self.config.EPOCHS,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=self._get_callbacks(),
            verbose=1
        )
        
        # Fine-tuning: unfreeze top layers
        print("\n🔓 Fine-tuning: unfreezing top layers...")
        for layer in self.model.layers:
            if hasattr(layer, 'trainable'):
                layer.trainable = True
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE * 0.1),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )
        
        finetune_callbacks = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
        ]
        
        self.model.fit(
            train_ds,
            epochs=20,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=finetune_callbacks,
            verbose=1
        )
        
        print("\n✅ Training complete!")
        return self.history
    
    def evaluate(self, test_ds, y_test):
        """Evaluate model on the test set and print detailed metrics."""
        print("\n📊 Evaluating model on test set...")
        loss, acc, top3 = self.model.evaluate(test_ds, verbose=0)
        
        print(f"   Test Loss:          {loss:.4f}")
        print(f"   Test Accuracy:      {acc*100:.2f}%")
        print(f"   Top-3 Accuracy:     {top3*100:.2f}%")
        
        # Predictions for detailed report
        y_pred_prob = self.model.predict(test_ds, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        class_names = [self.config.DISEASE_CLASSES[c] for c in self.config.CLASS_NAMES]
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return {
            'loss': float(loss),
            'accuracy': float(acc),
            'top3_accuracy': float(top3),
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
    
    def save_model(self, path=None):
        """Save the trained model."""
        path = path or self.config.MODEL_SAVE_PATH
        self.model.save(path)
        print(f"✅ Model saved to: {path}")
        
        # Save config
        cfg_path = path.replace('.h5', '_config.json')
        cfg_data = {
            'class_names': self.config.CLASS_NAMES,
            'disease_classes': self.config.DISEASE_CLASSES,
            'image_size': self.config.IMAGE_SIZE,
            'num_classes': self.config.NUM_CLASSES,
        }
        with open(cfg_path, 'w') as f:
            json.dump(cfg_data, f, indent=2)
        print(f"✅ Config saved to: {cfg_path}")
    
    def load_model(self, path=None):
        """Load a pre-trained model."""
        path = path or self.config.MODEL_SAVE_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}\nRun training first!")
        
        self.model = keras.models.load_model(path)
        print(f"✅ Model loaded from: {path}")
        return self.model


# ============================================================
# VISUALIZATION
# ============================================================

class Visualizer:
    """Plots training history, confusion matrix, etc."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def plot_training_history(self, history, save_path='training_history.png'):
        """Plot accuracy and loss curves."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        metrics = [
            ('accuracy', 'val_accuracy', 'Accuracy', 'Accuracy'),
            ('loss', 'val_loss', 'Loss', 'Loss'),
            ('top3_accuracy', 'val_top3_accuracy', 'Top-3 Accuracy', 'Top-3 Accuracy'),
        ]
        
        for ax, (train_key, val_key, title, ylabel) in zip(axes, metrics):
            if train_key in history.history:
                ax.plot(history.history[train_key], label='Train', color='#4CAF50', linewidth=2)
            if val_key in history.history:
                ax.plot(history.history[val_key], label='Validation', color='#2196F3', linewidth=2)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Training history plot saved: {save_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot normalized confusion matrix."""
        class_names = [self.config.DISEASE_CLASSES[c] for c in self.config.CLASS_NAMES]
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        for ax, data, title, fmt in [
            (ax1, cm, 'Raw Counts', 'd'),
            (ax2, cm_norm, 'Normalized', '.2f')
        ]:
            sns.heatmap(
                data, annot=True, fmt=fmt, ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues', linewidths=0.5
            )
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved: {save_path}")
    
    def plot_class_distribution(self, metadata, save_path='class_distribution.png'):
        """Bar chart of the class distribution."""
        dist = metadata['dx'].value_counts()
        names = [self.config.DISEASE_CLASSES.get(c, c) for c in dist.index]
        colors = [self.config.COLORS.get(c, '#888') for c in dist.index]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(dist)), dist.values, color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_xticks(range(len(dist)))
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_title('HAM10000 Class Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Images')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, dist.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(val), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Class distribution plot saved: {save_path}")


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    """Full end-to-end training pipeline."""
    print("=" * 70)
    print("🏥  AI SKIN DISEASE CLASSIFICATION - Training Pipeline")
    print("=" * 70)
    
    config = Config()
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # --- 1. Load Data ---
    loader = SkinDataLoader(config)
    train_ds, val_ds, test_ds, test_df = loader.split_and_build_datasets()
    
    viz = Visualizer(config)
    viz.plot_class_distribution(loader.metadata)
    
    # --- 2. Build Model ---
    skin_model = SkinDiseaseModel(config)
    skin_model.build_model_v1_efficient()
    
    # --- 3. Train ---
    history = skin_model.train(
        train_ds,
        val_ds,
        class_weights=loader.class_weights
    )
    
    # --- 4. Visualize Training ---
    viz.plot_training_history(history)
    
    # --- 5. Evaluate ---
    results = skin_model.evaluate(test_ds, test_df['label_idx'].values)
    viz.plot_confusion_matrix(test_df['label_idx'].values, results['y_pred'])
    
    # --- 6. Save Model ---
    skin_model.save_model()
    
    print("\n" + "=" * 70)
    print("🎉  Training Complete!")
    print(f"   Test Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"   Top-3 Accuracy: {results['top3_accuracy']*100:.2f}%")
    print(f"   Model saved to: {config.MODEL_SAVE_PATH}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    main()
