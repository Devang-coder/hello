import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import matplotlib.pyplot as plt

# Define constants
IMAGE_SIZE = (64, 64)  # Resize images to 64x64 to save memory
BATCH_SIZE = 8  # Reduced batch size to 8 for memory optimization
BASE_DIR = 'Data_Anuj'  # The main directory

# Function to preprocess image data
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)  # Normalize the image to [0,1]
    return image, label

# Function to get dataset from directory
def get_dataset(data_dir):
    return keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    ).map(process).cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE)

# Load and preprocess datasets (Train and Test directories)
train_ds = get_dataset(os.path.join(BASE_DIR, 'Train'))
validation_ds = get_dataset(os.path.join(BASE_DIR, 'Test'))

# Reduce the model complexity (fewer filters and units)
model = Sequential([
    Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model training with checkpoint to save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint('dysgraphia_model.h5',
                                                 save_best_only=True,
                                                 monitor='val_accuracy',
                                                 mode='max')

# Train the model (start with fewer epochs to reduce training time and memory usage)
history = model.fit(
    train_ds,
    epochs=5,  # Start with fewer epochs
    validation_data=validation_ds,
    callbacks=[checkpoint]  # Save the best model based on validation accuracy
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

# Function to classify a new image
def classify_image(image_path):
    img = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)  # Resize to 64x64
    img_array = keras.utils.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return (prediction[0][0] > 0.5).astype(int)  # Return predicted class

# Function to classify images in a directory
def classify_images_in_directory(directory):
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        if os.path.exists(img_path):
            predicted_class = classify_image(img_path)
            class_name = 'Dysgraphia' if predicted_class == 0 else 'Non-Dysgraphia'
            print(f"{img_name} is classified as {class_name}.")
        else:
            print(f"Image not found at: {img_path}")

# Test image paths (adjust path based on your test set structure)
dyslexic_folder = os.path.join('Data_Anuj', 'Test', 'dyslexic')
non_dyslexic_folder = os.path.join('Data_Anuj', 'Test', 'non_dyslexic')

# Classify images in 'Test/dyslexic' folder
print("Classifying images in 'Test/dyslexic' folder:")
classify_images_in_directory(dyslexic_folder)

# Classify images in 'Test/non_dyslexic' folder
print("\nClassifying images in 'Test/non_dyslexic' folder:")
classify_images_in_directory(non_dyslexic_folder)
