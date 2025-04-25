from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
app = Flask(__name__)
MODEL_PATH = 'dysgraphia_model.h5'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = load_model(MODEL_PATH)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        img = image.load_img(filepath, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
        result = "Dysgraphia" if prediction < 0.5 else "Non-Dysgraphia"
        return render_template('result.html',
                            result=result,
                            confidence=float(prediction),
                            image_path=filepath)
if __name__ == '__main__':
    app.run(debug=True)
