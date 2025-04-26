import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask app with explicit template/static folders
app = Flask(__name__, static_folder='static', template_folder='templates')

# ===== Configuration =====
MODEL_PATH = 'dysgraphia_model.h5'
UPLOAD_FOLDER = '/tmp/uploads'  # Changed to Vercel's tmp directory
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===== Model Loading =====
model = None

def init_model():
    """Safe model initializer with fallback for compatibility"""
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH, compile=False)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            app.logger.info("Model loaded with compile=False workaround")
        except Exception as e:
            app.logger.error(f"Model load failed: {str(e)}")
            raise

# ===== Static File Route =====
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# ===== Route Handlers =====
@app.route('/')
def home():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image predictions"""
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if not file or file.filename == '':
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "Invalid file type", 400
    if model is None:
        return "Service temporarily unavailable", 503

    try:
        # Secure file handling
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Prediction pipeline
        img = image.load_img(filepath, target_size=(64, 64))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        result = "Dysgraphia" if prediction < 0.5 else "Non-Dysgraphia"

        return render_template(
            'result.html',
            result=result,
            confidence=float(prediction),
            image_path=filename
        )

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return "Error processing image", 500

# ===== Helper Functions =====
def allowed_file(filename):
    """Check for allowed file extensions"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===== Initialization =====
init_model()  # Initialize immediately for Vercel

# Vercel requires this exact variable name
# THIS MUST BE THE LAST LINE IN THE FILE
handler = app
