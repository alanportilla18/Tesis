import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_type="cnn", model_path=None):
        if model_type == "cnn":
            default_path = "models/emotion_cnn_model.h5"
        else:  # resnet
            default_path = "models/emotion_model.h5"
        
        self.model_path = model_path if model_path else default_path
        self.model = load_model(self.model_path)
        self.emotion_labels = ["angry", "happy", "neutral", "sad", "disgust"]
        self.target_size = (48, 48)  # Tamaño de entrada para ambos modelos

    def preprocess_image(self, image):
        """Preprocesa la imagen para el modelo."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        processed = np.expand_dims(normalized, axis=(0, -1))
        return processed

    def predict_emotion(self, image):
        """Predice la emoción de una imagen."""
        try:
            processed_image = self.preprocess_image(image)
            prediction = self.model.predict(processed_image)
            emotion_idx = np.argmax(prediction, axis=1)[0]
            emotion_label = self.emotion_labels[emotion_idx]
            return emotion_label
        except Exception as e:
            print(f"Error al predecir emoción: {e}")
            return "unknown"