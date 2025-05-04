Emotion Recognition System
Sistema de reconocimiento facial y detección de emociones en tiempo real.
Estructura

data/: Dataset y rostros conocidos.
models/: Modelos entrenados.
outputs/: Reportes generados.
src/: Código fuente.
scripts/: Scripts de entrenamiento y evaluación.

Instalación

Crear entorno virtual:
python -m venv venv
source venv/bin/activate


Instalar dependencias:
pip install -r requirements.txt


Configurar dataset en data/dataset/ y rostros en data/rostros/.


Uso

Entrenar modelo:
python scripts/train_model.py


Ejecutar sistema:
python src/main.py



Requisitos

Python 3.8+
Cámara web
Dataset de emociones (ej., FER-2013)

