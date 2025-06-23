import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import cv2
import base64
import json

app = Flask(__name__)

json_path = os.path.join('Data', 'chihuahuavsmuffin.json')
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Archivo JSON del modelo no encontrado en {json_path}")

weights_path = os.path.join('Data', 'chihuahuavsmuffin.h5')
if not os.path.exists(weights_path):
    weights_path = os.path.join(os.getcwd(), 'chihuahuavsmuffin.h5')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pesos del modelo no encontrados en {weights_path}")

try:
    with open(json_path, 'r') as json_file:
        model_json = json_file.read()
        
    model = tf.keras.models.model_from_json(model_json)
    
    model.load_weights(weights_path)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    print("Modelo cargado correctamente desde JSON y pesos")
    
except Exception as e:
    print(f"Error al cargar el modelo desde JSON: {e}")
    print("Intentando cargar modelo completo desde archivo .h5...")
    
    try:
        model = tf.keras.models.load_model(weights_path)
        print("Modelo completo cargado correctamente desde archivo .h5")
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {e}")

IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_NAMES = ["chihuahua", "muffin"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    
    image_data = request.json['image']
    image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img, verbose=0)
    class_idx = int(np.argmax(prediction[0]))  # Convertir a entero de Python
    confidence = float(prediction[0][class_idx] * 100)  # Convertir a float de Python
    
    return jsonify({
        'class': CLASS_NAMES[class_idx],
        'confidence': confidence,
        'wearing_chihuahuavsmuffin': bool(class_idx == 0)  # Convertir a booleano de Python
    })

if __name__ == '__main__':
    app.run(debug=True)