import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# ===== Configuration =====
MODEL_PATH = 'dysgraphia_model.h5'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONUNBUFFERED'] = '1'  # Better logging in Railway

# ===== Model Loading =====
model = None

def init_model():
    """Safe model initializer with fallback for compatibility"""
    global model
    if model is None:
        try:
            # Attempt normal load first
            model = load_model(MODEL_PATH)
            app.logger.info("Model loaded with original configuration")
        except Exception as e:
            app.logger.warning(f"Standard load failed: {str(e)}")
            try:
                # Fallback for optimizer compatibility
                model = load_model(MODEL_PATH, compile=False)
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                app.logger.info("Model loaded via compile=False workaround")
            except Exception as e:
                app.logger.error(f"Critical model load failure: {str(e)}")
                raise

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
if os.environ.get('RAILWAY_ENVIRONMENT'):
    # Production - load immediately
    init_model()
else:
    # Development - lazy load
    @app.before_first_request
    def lazy_load_model():
        init_model()

if __name__ == '__main__':
    init_model()  # Ensure model loads before first request
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )
