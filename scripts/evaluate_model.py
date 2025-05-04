import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

# Configuración
DATA_DIR = "data/dataset"
TEST_DIR = os.path.join(DATA_DIR, "test")
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EMOTIONS = ["angry", "happy", "neutral", "sad", "disgust"]
MODEL_PATHS = {
    "resnet": "models/emotion_model.h5",
    "cnn": "models/emotion_cnn_model.h5"
}

def create_test_generator():
    """Crea un generador para el conjunto de test."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=False
    )
    return test_generator

def evaluate_model(model_path, model_name, test_generator):
    """Evalúa un modelo y muestra métricas."""
    print(f"\nEvaluando modelo: {model_name}")
    model = load_model(model_path)
    
    # Evaluar en el conjunto de test
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predicciones
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Reporte de clasificación
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=EMOTIONS))
    
    # Matriz de confusión
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_classes)
    print(cm)

def main():
    # Crear generador de test
    test_generator = create_test_generator()
    
    # Evaluar ambos modelos
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            evaluate_model(model_path, model_name, test_generator)
        else:
            print(f"Modelo {model_name} no encontrado en: {model_path}")

if __name__ == "__main__":
    main()