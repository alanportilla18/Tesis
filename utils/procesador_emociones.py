import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Cargar modelo
modelo_emociones = load_model("modelo/modelFEC.h5")

# Etiquetas de emociones (según tu entrenamiento)
etiquetas_emociones = ['angry','disgust','happy','neutral','sad']

def preprocesar_rostro(img_rostro):
    """
    Recibe un rostro (imagen RGB), lo convierte a escala de grises,
    lo redimensiona a 48x48 y lo normaliza.
    """
    rostro_gris = cv2.cvtColor(img_rostro, cv2.COLOR_BGR2GRAY)
    rostro_redim = cv2.resize(rostro_gris, (48, 48))
    rostro_normalizado = rostro_redim / 255.0
    rostro_reshaped = np.reshape(rostro_normalizado, (1, 48, 48, 1))
    return rostro_reshaped

def predecir_emocion(img_rostro):
    """
    Recibe una imagen de rostro y devuelve la emoción predicha.
    """
    entrada = preprocesar_rostro(img_rostro)
    prediccion = modelo_emociones.predict(entrada, verbose=0)
    indice_emocion = np.argmax(prediccion)
    return etiquetas_emociones[indice_emocion]
