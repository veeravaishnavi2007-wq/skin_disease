"""
Flask Web API for Skin Disease Classification
==============================================
Serves a beautiful web interface and REST API endpoints.

Usage:
    python app.py

Then open: http://localhost:5000
"""

import os
import sys
import io
import json
import time
import base64
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from functools import wraps

# Fix Windows console encoding issues for emojis and block characters
if sys.platform.startswith('win'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, abort
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Lazy-import heavy ML libraries
_engine = None

def get_engine():
    """Lazy-load the prediction engine."""
    global _engine
    if _engine is None:
        try:
            from predict_disease import SkinDiseasePredictionEngine
            model_path = os.environ.get('MODEL_PATH', './skin_disease_model.h5')
            _engine = SkinDiseasePredictionEngine(model_path)
        except Exception as e:
            print(f"⚠️  Could not load prediction engine: {e}")
            _engine = None
    return _engine


# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)
CORS(app)

app.config.update(
    SECRET_KEY=os.urandom(24),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,   # 16 MB max upload
    UPLOAD_FOLDER=tempfile.gettempdir(),
)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def error_response(message: str, status: int = 400):
    return jsonify({'success': False, 'error': message}), status


# ============================================================
# ROUTES - Pages
# ============================================================

@app.route('/')
def index():
    """Serve the main web application."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    engine = get_engine()
    return jsonify({
        'status':       'ok',
        'model_loaded': engine is not None and engine.model is not None,
        'timestamp':    datetime.now().isoformat(),
    })


# ============================================================
# ROUTES - API
# ============================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Accepts: multipart/form-data with 'image' file OR JSON with 'image_base64' field.
    Returns: JSON prediction result.
    """
    engine = get_engine()
    
    if engine is None or engine.model is None:
        return error_response(
            'Model not loaded. Please train the model first (run skin_disease_classifier.py).', 503
        )
    
    start_time = time.time()
    
    try:
        # ---- Multipart file upload ----
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return error_response('No file selected')
            if not allowed_file(file.filename):
                return error_response(f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}')
            
            image_bytes = file.read()
            filename = secure_filename(file.filename)
        
        # ---- Base64 JSON ----
        elif request.is_json and 'image_base64' in request.json:
            data = request.json['image_base64']
            if ',' in data:          # strip data URI prefix
                data = data.split(',', 1)[1]
            image_bytes = base64.b64decode(data)
            filename = request.json.get('filename', 'uploaded_image.jpg')
        
        else:
            return error_response('No image provided. Send "image" as file upload or "image_base64" in JSON.')
        
        # ---- Predict ----
        result = engine.predict_from_bytes(image_bytes, filename=filename)
        
        elapsed = round((time.time() - start_time) * 1000, 1)
        
        # Build clean response
        response = {
            'success':               True,
            'inference_time_ms':     elapsed,
            'filename':              filename,
            'predicted_disease':     result['predicted_disease_name'],
            'predicted_code':        result['predicted_disease_code'],
            'confidence':            result['confidence_percentage'],
            'is_critical':           result['is_critical'],
            'disease_info': {
                'name':        result['disease_info']['name'],
                'severity':    result['disease_info']['severity'],
                'risk_level':  result['disease_info']['risk_level'],
                'description': result['disease_info']['description'],
                'symptoms':    result['disease_info']['symptoms'],
                'treatment':   result['disease_info']['treatment'],
                'action':      result['disease_info']['action'],
                'color':       result['disease_info']['color'],
                'emoji':       result['disease_info']['emoji'],
            },
            'all_predictions': [
                {
                    'code':       code,
                    'name':       data['name'],
                    'confidence': data['percentage'],
                    'color':      data['color'],
                    'emoji':      data['emoji'],
                }
                for code, data in result['all_predictions'].items()
            ],
        }
        
        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()
        return error_response(f'Prediction failed: {str(e)}', 500)


@app.route('/api/diseases', methods=['GET'])
def diseases():
    """Return information about all supported diseases."""
    from predict_disease import DISEASE_INFO
    return jsonify({
        'success':  True,
        'diseases': list(DISEASE_INFO.values())
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model metadata."""
    engine = get_engine()
    
    if engine is None or engine.model is None:
        return jsonify({
            'success':      True,
            'model_loaded': False,
            'message':      'Model not trained yet. Run skin_disease_classifier.py first.',
        })
    
    try:
        total_params = engine.model.count_params()
    except Exception:
        total_params = None
    
    return jsonify({
        'success':       True,
        'model_loaded':  True,
        'model_path':    engine.model_path,
        'total_params':  total_params,
        'num_classes':   7,
        'image_size':    224,
        'architecture':  'EfficientNetB2',
        'dataset':       'HAM10000',
    })


# ============================================================
# STATIC FILES
# ============================================================

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size: 16MB'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🏥  Skin Disease AI - Web Application")
    print("=" * 60)
    print(f"   Serving on: http://localhost:5000")
    print(f"   API docs:   http://localhost:5000/health")
    print(f"   Model:      {os.environ.get('MODEL_PATH', './skin_disease_model.h5')}")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
    )
