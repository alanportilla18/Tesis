import numpy as np
import cv2
import tensorflow as tf
import os

# Ruta al modelo exportado
ruta_modelo = r'C:\Users\porti\Documents\Tesis\modelFEC'
modelo_emocion = tf.keras.models.load_model(ruta_modelo)

# Extraer la función de inferencia desde el modelo
modelo_emocion_serving = modelo_emocion.signatures["serving_default"]

emociones = ['angry', 'disgust', 'happy', 'neutral', 'sad']

def predecir_emocion(rostro):
    rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
    rostro_redimensionado = cv2.resize(rostro_gray, (48, 48))
    input_array = np.expand_dims(rostro_redimensionado, axis=(0, -1)) / 255.0
    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)

    # Ejecutar la inferencia
    resultado = modelo_emocion_serving(input_tensor)

    # Obtener el nombre de salida dinámicamente
    output_key = list(resultado.keys())[0]
    prediccion = resultado[output_key].numpy()

    emocion_idx = np.argmax(prediccion)
    return emociones[emocion_idx]
