import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 8
EPOCHS = 10
BASE_DIR = 'Data_Anuj'

# 1. Data Loading
def load_dataset(data_dir, subset=None):
    return keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels='inferred',
        label_mode='binary',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        validation_split=0.2 if subset == 'validation' else None,
        subset=subset,
        seed=42
    ).map(lambda x, y: (x/255.0, y)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 2. Model Architecture
def create_model():
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall')])
    return model

# 3. Training Function
def train_model():
    train_ds = load_dataset(os.path.join(BASE_DIR, 'Train'))
    val_ds = load_dataset(os.path.join(BASE_DIR, 'Train'), subset='validation')

    model = create_model()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    return model

# 4. TFLite Conversion
def convert_to_tflite():
    model = keras.models.load_model('best_model.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('dysgraphia_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted to TFLite format")

# 5. Evaluation Class
class DysgraphiaEvaluator:
    def __init__(self, model_path='best_model.keras'):
        self.model_path = model_path
        self.is_tflite = model_path.endswith('.tflite')

        if self.is_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = keras.models.load_model(model_path)

    def predict_image(self, image_path, threshold=0.5):
        img = image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        if self.is_tflite:
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                img_array.astype(np.float32)
            )
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(
                self.output_details[0]['index'])[0][0]
        else:
            prediction = self.model.predict(img_array)[0][0]

        label = 'Dysgraphia' if prediction < threshold else 'Non-Dysgraphia'
        return label, float(prediction)

    def evaluate_test_set(self, test_dir):
        class_names = ['dyslexic', 'non_dyslexic']
        results = {class_name: {'correct': 0, 'total': 0}
                 for class_name in class_names}

        for class_name in class_names:
            class_dir = os.path.join(test_dir, class_name)
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                true_label = 'Dysgraphia' if class_name == 'dyslexic' else 'Non-Dysgraphia'

                try:
                    pred_label, confidence = self.predict_image(img_path)
                    results[class_name]['total'] += 1
                    if pred_label == true_label:
                        results[class_name]['correct'] += 1
                    print(f"{img_file}: Predicted {pred_label} ({confidence:.2%}) | Actual {true_label}")
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")

        print("\nEvaluation Summary:")
        for class_name in class_names:
            acc = results[class_name]['correct'] / results[class_name]['total']
            print(f"{class_name}: {results[class_name]['correct']}/{results[class_name]['total']} ({acc:.2%})")

# Main Execution
if __name__ == '__main__':
    # Train if no model exists
    if not os.path.exists('best_model.keras'):
        print("Training new model...")
        train_model()

    # Convert to TFLite (optional)
    if not os.path.exists('dysgraphia_model.tflite'):
        print("\nConverting to TFLite format...")
        convert_to_tflite()

    # Evaluate with Keras model
    print("\nEvaluating with Keras model:")
    keras_evaluator = DysgraphiaEvaluator('best_model.keras')
    keras_evaluator.evaluate_test_set(os.path.join(BASE_DIR, 'Test'))

    # Evaluate with TFLite model
    print("\nEvaluating with TFLite model:")
    tflite_evaluator = DysgraphiaEvaluator('dysgraphia_model.tflite')
    tflite_evaluator.evaluate_test_set(os.path.join(BASE_DIR, 'Test'))
