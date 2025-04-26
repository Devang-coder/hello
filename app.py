import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
MODEL_PATH = 'dysgraphia_model.h5'
UPLOAD_FOLDER = '/tmp/uploads'  # Vercel-compatible temp storage
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model Loading
model = None

def init_model():
    """Optimized model loader for TF 2.6/Python 3.8"""
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH, compile=False)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

# Static files route
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Routes (keep your existing route handlers)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # [Keep your existing predict() function unchanged]
    pass

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize model immediately
init_model()

# Vercel requirement - MUST BE LAST LINE
handler = app
