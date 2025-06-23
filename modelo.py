import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import kagglehub
import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Download latest version
path = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")

print("Path to dataset files:", path)

# Constants for improved model
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 2

# Clean dataset by removing corrupt images and non-jpg files
print("Cleaning dataset...")
MAIN_PATH = os.path.join(path, "test")
corrupt_files = []

# Identify corrupt images
for content in ('chihuahua', 'muffin'):
    category_path = os.path.join(MAIN_PATH, content)
    for file in os.listdir(category_path):
        file_path = os.path.join(category_path, file)
        try:
            # Check if image can be opened
            img = PIL.Image.open(file_path)
            img.verify()  # Verify it's a valid image

            # Remove non-jpg files
            if not file.lower().endswith('.jpg'):
                os.remove(file_path)
                print(f"Removed non-jpg file: {file_path}")
        except (PIL.UnidentifiedImageError, OSError, Exception) as e:
            corrupt_files.append(file_path)
            print(f"Corrupt image found: {file_path}")

# Remove corrupt files
for file_path in corrupt_files:
    try:
        os.remove(file_path)
        print(f"Removed corrupt file: {file_path}")
    except:
        print(f"Failed to remove: {file_path}")

train_path = os.path.join(path, "train")
test_path = os.path.join(path, "test")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_ds = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
# Show sample images from the dataset
print("Displaying sample images...")
plt.figure(figsize=(20, 10))
images, labels = next(train_ds)
for i in range(min(15, len(images))):
    plt.subplot(3, 5, i+1)
    plt.imshow(images[i])
    plt.title('Chihuahua' if np.argmax(labels[i]) == 0 else 'Muffin')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Crear el transfer learning
print("Modelo DenseNet201...")
# Initializar el DenseNet201 con pesos preentrenados
base_model = DenseNet201(include_top=False,
                         weights='imagenet',
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Congelar todas las capas en el modelo base
for layer in base_model.layers:
    layer.trainable = False

# customizar el modelo
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)  # Add dropout to reduce overfitting
x = tf.keras.layers.Dense(50, activation='relu')(x)
predictions = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

# Crear el modelo final
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
def create_callbacks():
    checkpoint_path = './best_model.h5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        mode='max',
        factor=0.1,
        patience=3,
        verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=10,
        verbose=1
    )
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='logs/densenet_cats_dogs',
        histogram_freq=1
    )
    callbacks = [checkpoint, reduce_lr, early_stop, tensorboard]
    return callbacks

# Compilar el modelo con learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['accuracy']
)


# entrenar el moddelo con las mejoras
print("Entrenando el modelo...")
mis_callbacks = create_callbacks()

history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=mis_callbacks
)

# Evaluate the model
print("Evaluando el modelo...")
evaluation = model.evaluate(val_ds)
print(f"Validation Loss: {evaluation[0]:.4f}")
print(f"Validation Accuracy: {evaluation[1]:.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validación', linewidth=2)
plt.title('Exactitud del modelo')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
plt.plot(history.history['val_loss'], label='Validación', linewidth=2)
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Guardar el modelo
model_json = model.to_json()
with open("chihuahuavsmuffin.json", "w") as json_file:
    json_file.write(model_json)
model.save("chihuahuavsmuffin.h5")
print("Modelo guardado como chihuahuavsmuffin.json y chihuahuavsmuffin.h5")