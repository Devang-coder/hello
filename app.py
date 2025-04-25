from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Set model path and upload folder
MODEL_PATH = 'dysgraphia_model.h5'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = load_model(MODEL_PATH)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get prediction from the model
        prediction = model.predict(img_array)[0][0]

        # Determine result
        result = "Dysgraphia" if prediction < 0.5 else "Non-Dysgraphia"

        return render_template('result.html',
                               result=result,
                               confidence=float(prediction),
                               image_path=filepath)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
