import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import matplotlib.pyplot as plt

# Further reduce image size to 64x64 to save memory
IMAGE_SIZE = (64, 64)  # Resize images to 64x64 to save more memory
BATCH_SIZE = 16
BASE_DIR = 'Data_Anuj'  # The main directory

# Load and preprocess datasets (Train and Test directories)
train_ds = keras.utils.image_dataset_from_directory(
    directory=os.path.join(BASE_DIR, 'Train'),
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory=os.path.join(BASE_DIR, 'Test'),
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

# Normalize the images to scale the pixel values between 0 and 1
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)  # Normalize the image to [0,1]
    return image, label

# Map, cache, and prefetch for optimized pipeline
train_ds = train_ds.map(process).cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.map(process).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Reduce the model complexity (fewer filters and units)
model = Sequential([
    Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),  # Fewer filters
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(3, 3), activation='relu'),  # Fewer filters
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(16, activation='relu'),  # Fewer units in Dense layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification (Dysgraphia/Non-Dysgraphia)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model (start with fewer epochs for quicker training and less memory)
history = model.fit(
    train_ds,
    epochs=5,  # Start with fewer epochs to reduce training time and memory usage
    validation_data=validation_ds
)

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], color='red', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model
model.save('dysgraphia_model.h5')
print("Model saved as 'dysgraphia_model.h5'")

# Function to classify a new image
def classify_image(image_path):
    # Ensure the image is resized to 64x64 during inference
    img = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)  # Resize to 64x64
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    prediction = model.predict(img_array)
    class_label = (prediction[0][0] > 0.5).astype(int)  # Threshold the prediction at 0.5
    return class_label

# Test image paths (adjust path based on your test set structure)
dyslexic_folder = os.path.join('Data_Anuj', 'Test', 'dyslexic')
non_dyslexic_folder = os.path.join('Data_Anuj', 'Test', 'non_dyslexic')

# Iterate over all images in 'Test/dyslexic' folder
print("Classifying images in 'Test/dyslexic' folder:")
for img_name in os.listdir(dyslexic_folder):
    img_path = os.path.join(dyslexic_folder, img_name)
    if os.path.exists(img_path):
        predicted_class = classify_image(img_path)
        if predicted_class == 0:
            print(f"{img_name} is classified as Dysgraphia.")
        else:
            print(f"{img_name} is classified as Non-Dysgraphia.")
    else:
        print(f"Image not found at: {img_path}")

# Iterate over all images in 'Test/non_dyslexic' folder
print("\nClassifying images in 'Test/non_dyslexic' folder:")
for img_name in os.listdir(non_dyslexic_folder):
    img_path = os.path.join(non_dyslexic_folder, img_name)
    if os.path.exists(img_path):
        predicted_class = classify_image(img_path)
        if predicted_class == 0:
            print(f"{img_name} is classified as Dysgraphia.")
        else:
            print(f"{img_name} is classified as Non-Dysgraphia.")
    else:
        print(f"Image not found at: {img_path}")
