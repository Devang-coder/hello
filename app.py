import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import tempfile
import logging

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
MODEL_PATH = os.path.abspath('dysgraphia_model.tflite')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.info(f"Initializing with model at: {MODEL_PATH}")

class DysgraphiaPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        with open(MODEL_PATH, 'rb') as f:
            self.interpreter = tf.lite.Interpreter(model_content=f.read())
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        app.logger.info("Model loaded successfully")

    def predict(self, img_array):
        img_array = img_array.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

# Initialize predictor
predictor = None
try:
    predictor = DysgraphiaPredictor()
except Exception as e:
    app.logger.error(f"Predictor initialization failed: {str(e)}")

@app.before_request
def validate_request():
    if request.method == 'POST' and 'file' in request.files:
        if request.content_length > MAX_FILE_SIZE:
            return jsonify({'error': f'File exceeds {MAX_FILE_SIZE//1024//1024}MB limit'}), 400

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({'error': 'Prediction service unavailable'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Allowed file types: png, jpg, jpeg'}), 400

    try:
        ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            try:
                img = tf.keras.preprocessing.image.load_img(tmp.name, target_size=(64, 64))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = predictor.predict(img_array)
                result = 'Dysgraphia Detected' if prediction > 0.5 else 'No Dysgraphia Detected'
                confidence = round((prediction if prediction > 0.5 else 1 - prediction) * 100, 2)

                return jsonify({
                    'result': result,
                    'confidence': confidence,
                    'model': 'tflite',
                    'version': '1.0'
                })
            finally:
                os.unlink(tmp.name)
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Processing failed. Please try another image.'}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Vercel requirement
handler = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
