import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory, jsonify
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Secure filename and save temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize

            # Make prediction
            prediction = model.predict(img_array)
            result = 'Dysgraphia Detected' if prediction[0][0] > 0.5 else 'No Dysgraphia Detected'
            confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])

            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                'result': result,
                'confidence': round(confidence * 100, 2)
            })

        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize model immediately
init_model()

# Vercel requirement - MUST BE LAST LINE
handler = app
