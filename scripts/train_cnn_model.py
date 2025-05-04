import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Configuración
DATA_DIR = "data/dataset"  # Ruta al dataset
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMG_SIZE = (48, 48)  # Tamaño de las imágenes
BATCH_SIZE = 64
NUM_CLASSES = 5  # Emociones: angry, happy, neutral, sad, disgust
EPOCHS = 50
MODEL_PATH = "models/emotion_cnn_model.h5"

# Definir emociones
EMOTIONS = ["angry", "happy", "neutral", "sad", "disgust"]

def create_data_generators():
    """Crea generadores de datos con aumento para entrenamiento y validación."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=False
    )
    
    return train_generator, test_generator

def build_cnn_model():
    """Construye el modelo CNN."""
    model = Sequential([
        # Bloque 1
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Bloque 2
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Bloque 3
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Bloque 4
        Conv2D(512, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Capas densas
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    
    return model

def plot_training_history(history):
    """Grafica la precisión y pérdida del entrenamiento."""
    plt.figure(figsize=(12, 4))
    
    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.savefig("outputs/cnn_training_history.png")
    plt.close()

def main():
    # Crear generadores de datos
    train_generator, test_generator = create_data_generators()
    
    # Construir y compilar el modelo
    model = build_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy")
    ]
    
    # Entrenar el modelo
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=callbacks
    )
    
    # Graficar historial de entrenamiento
    plot_training_history(history)
    
    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predicciones para métricas detalladas
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Reporte de clasificación
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=EMOTIONS))
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()