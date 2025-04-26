import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
import matplotlib.pyplot as plt
import h5py
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 8  # Small batch size for limited memory
EPOCHS = 10  # Increased epochs for better learning
BASE_DIR = 'Data_Anuj'  # Main data directory

# Data preprocessing function
def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # More efficient normalization
    return image, label

# Dataset loader with improved settings
def load_dataset(data_dir, subset=None):
    return keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels='inferred',
        label_mode='binary',  # Better for binary classification
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        validation_split=0.2 if subset == 'validation' else None,
        subset=subset,
        seed=42
    ).map(preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Load datasets with validation split
train_ds = load_dataset(os.path.join(BASE_DIR, 'Train'))
val_ds = load_dataset(os.path.join(BASE_DIR, 'Train'), subset='validation')
test_ds = load_dataset(os.path.join(BASE_DIR, 'Test'))

# Enhanced model architecture
def create_model():
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(*IMAGE_SIZE, 3)),  # Built-in normalization

        # Feature extraction
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        # Classification
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),  # Regularization
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall')])
    return model

model = create_model()

# Callbacks for better training
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint('best_model.keras',
                                                 save_best_only=True,
                                                 monitor='val_accuracy')

# Training with validation
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluation on test set
print("\nEvaluating on test set:")
test_results = model.evaluate(test_ds)
print(f"Test Accuracy: {test_results[1]:.2%}")
print(f"Test Precision: {test_results[2]:.2%}")
print(f"Test Recall: {test_results[3]:.2%}")

# Visualization
def plot_training(history):
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

    plt.tight_layout()
    plt.show()

plot_training(history)

# Prediction functions
def predict_image(image_path, model, threshold=0.5):
    img = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    prediction = model.predict(img_array)[0][0]
    return 'Dysgraphia' if prediction < threshold else 'Non-Dysgraphia', prediction

def evaluate_test_set(test_dir, model):
    class_names = ['dyslexic', 'non_dyslexic']
    results = {class_name: {'correct': 0, 'total': 0} for class_name in class_names}

    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            true_label = 'Dysgraphia' if class_name == 'dyslexic' else 'Non-Dysgraphia'

            pred_label, confidence = predict_image(img_path, model)
            results[class_name]['total'] += 1
            if pred_label == true_label:
                results[class_name]['correct'] += 1

            print(f"{img_file}: Predicted {pred_label} ({confidence:.2%}) | Actual {true_label}")

    # Print summary
    print("\nEvaluation Summary:")
    for class_name in class_names:
        acc = results[class_name]['correct'] / results[class_name]['total']
        print(f"{class_name}: {results[class_name]['correct']}/{results[class_name]['total']} ({acc:.2%})")

# Load best saved model
best_model = keras.models.load_model('best_model.keras')

# Evaluate on test set
test_dir = os.path.join(BASE_DIR, 'Test')
print("\nRunning evaluation on test set:")
evaluate_test_set(test_dir, best_model)
