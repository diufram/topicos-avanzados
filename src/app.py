from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Cargar el modelo
model = tf.keras.models.load_model('perros-gatos-cnn-ad.h5')



# Paso 1: Definir la función de preprocesamiento
def preprocess_image(image):
    image = image.resize((100, 100))
    image_array = np.array(image)
    if len(image_array.shape) == 2:  # Imagen en escala de grises
        image_array = np.expand_dims(image_array, axis=-1)  # Convertir a forma (100, 100, 1)
    else:
        image_array = np.expand_dims(image_array, axis=-1)  # Asegurarse de que tenga un solo canal
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Añadir la dimensión del batch
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Abre la imagen
        image = Image.open(file.stream).convert('L')  # Convertir directamente a escala de grises
        # Preprocesa la imagen
        image_array = preprocess_image(image)
        # Realiza la predicción
        prediction = model.predict(image_array)
        class_index = int(prediction[0] > 0.5)  # Umbral de 0.5 para clasificación binaria
        
        # Mapear el índice a nombre de clase
        class_names = ['cats', 'dogs']  # Asegúrate de que coincida con el orden en el entrenamiento
        class_name = class_names[class_index]
        
        return jsonify({'class_index': class_index, 'class_name': class_name}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
