import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import matplotlib.pyplot as plt
BASE_DIR = 'Data_Anuj'
train_ds = keras.utils.image_dataset_from_directory(
    directory=os.path.join(BASE_DIR, 'train'),
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)
validation_ds = keras.utils.image_dataset_from_directory(
    directory=os.path.join(BASE_DIR, 'Test'),
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=validation_ds
)
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], color='red', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
model.save('dysgraphia_model.h5')
print("Model saved as 'dysgraphia_model.h5'")
def classify_image(image_path):
    img = keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    prediction = model.predict(img_array)
    class_label = (prediction[0][0] > 0.5).astype(int)
    return class_label
test_image_path = os.path.join('test_images', 'sample.jpg')
if os.path.exists(test_image_path):
    predicted_class = classify_image(test_image_path)
    if predicted_class == 0:
        print("The image is classified as Dysgraphia.")
    else:
        print("The image is classified as Non-Dysgraphia.")
else:
    print(f"Test image not found at: {test_image_path}")
