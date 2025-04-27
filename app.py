import os
import numpy as np  # Make sure numpy is imported
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dynamically find the model path
MODEL_PATH = os.path.join(os.getcwd(), 'dysgraphia_model.tflite')
print(MODEL_PATH)
# Load the TFLite model
class Predictor:
    def __init__(self, model_path):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def predict(self, img_array):
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0][0]

predictor = Predictor(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
#
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#
#         # Create folder if doesn't exist
#         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#         file.save(file_path)
#
#         try:
#             img = tf.keras.preprocessing.image.load_img(file_path, target_size=(64, 64))
#             img_array = tf.keras.preprocessing.image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0) / 255.0
#
#             prediction = predictor.predict(img_array)
#             result = 'Dysgraphia Detected' if prediction > 0.5 else 'No Dysgraphia Detected'
#             confidence = round((prediction if prediction > 0.5 else 1 - prediction) * 100, 2)
#
#             return render_template('result.html',
#                                    result=result,
#                                    confidence=confidence / 100,
#                                    image_path='/' + file_path)
#         except Exception as e:
#             return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
#
#     else:
#         return jsonify({'error': 'Allowed file types: png, jpg, jpeg'}), 400

if __name__ == '__main__':
    app.run(debug=True)
