import os
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm
from mtcnn import MTCNN
import cv2

class FaceRecognizer:
    def __init__(self, rostros_dir="data/rostros", model_name="VGG-Face", umbral_confianza=9.0):
        self.rostros_dir = rostros_dir
        self.model_name = model_name
        self.umbral_confianza = umbral_confianza
        self.database = {}
        self.embeddings = {}
        self.detector = MTCNN()
        self.load_faces()

    def load_faces(self):
        """Carga rostros conocidos y genera embeddings."""
        if not os.path.exists(self.rostros_dir):
            raise FileNotFoundError(f"Carpeta {self.rostros_dir} no encontrada")
        
        print(f"Cargando rostros desde: {self.rostros_dir}")
        archivos = [f for f in os.listdir(self.rostros_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"Archivos encontrados: {archivos}")
        
        for archivo in archivos:
            nombre = os.path.splitext(archivo)[0]
            ruta = os.path.join(self.rostros_dir, archivo)
            self.database[nombre] = ruta
            print(f"Procesando: {nombre} ({ruta})")
        
        for nombre, ruta in self.database.items():
            try:
                img = cv2.imread(ruta)
                if img is None:
                    print(f"[ERROR] No se pudo cargar la imagen: {ruta}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(
                    img_path=img_rgb,
                    model_name=self.model_name,
                    detector_backend='mtcnn',
                    enforce_detection=False
                )
                print(f"Resultado de DeepFace.represent para {nombre}: {result[:10]} ... (longitud: {len(result)})")
                
                # Manejar diferentes estructuras de salida
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and "embedding" in result[0]:
                        embedding = result[0]["embedding"]
                    elif isinstance(result[0], float):
                        # Caso en que result es directamente el embedding
                        embedding = result
                    else:
                        print(f"[ERROR] Estructura inesperada en DeepFace.represent: {result}")
                        continue
                elif isinstance(result, dict) and "embedding" in result:
                    embedding = result["embedding"]
                else:
                    print(f"[ERROR] Estructura inesperada en DeepFace.represent: {result}")
                    continue
                
                self.embeddings[nombre] = embedding
                print(f"✓ Embedding generado para: {nombre} (longitud: {len(embedding)})")
            except Exception as e:
                print(f"[ADVERTENCIA] No se pudo procesar {nombre}: {e}")

        print(f"Se cargaron {len(self.embeddings)} rostros en la base de datos.")

    def detect_faces(self, frame):
        """Detecta rostros en un frame usando MTCNN."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(frame_rgb)
        print(f"Rostros detectados en el frame: {len(faces)}")
        return [
            {
                "facial_area": {
                    "x": int(face["box"][0]),
                    "y": int(face["box"][1]),
                    "w": int(face["box"][2]),
                    "h": int(face["box"][3])
                }
            }
            for face in faces
        ]

    def recognize_face(self, rostro):
        """Identifica un rostro y devuelve el nombre y la confianza."""
        try:
            rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
            result = DeepFace.represent(
                img_path=rostro_rgb,
                model_name=self.model_name,
                detector_backend='mtcnn',
                enforce_detection=False
            )
            print(f"Resultado de DeepFace.represent en recognize_face: {result[:10]} ... (longitud: {len(result)})")
            
            # Manejar diferentes estructuras de salida
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "embedding" in result[0]:
                    embedding = result[0]["embedding"]
                elif isinstance(result[0], float):
                    # Caso en que result es directamente el embedding
                    embedding = result
                else:
                    print(f"[ERROR] Estructura inesperada en DeepFace.represent: {result}")
                    return "Error", "N/A", float('inf')
            elif isinstance(result, dict) and "embedding" in result:
                embedding = result["embedding"]
            else:
                print(f"[ERROR] Estructura inesperada en DeepFace.represent: {result}")
                return "Error", "N/A", float('inf')
            
            nombre = "Desconocido"
            menor_distancia = float('inf')
            confianza = "N/A"

            for persona, emb in self.embeddings.items():
                distancia = norm(np.array(embedding) - np.array(emb))
                print(f"Comparando con {persona}: distancia = {distancia:.2f}")
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    if distancia < self.umbral_confianza:
                        nombre = persona
                        confianza = f"{max(0, 100 - (distancia * 10)):.1f}%"
                    else:
                        nombre = "Desconocido"
                        confianza = "N/A"

            print(f"Resultado: {nombre}, distancia mínima: {menor_distancia:.2f}, confianza: {confianza}")
            return nombre, confianza, menor_distancia
        except Exception as e:
            print(f"Error al reconocer rostro: {e}")
            return "Error", "N/A", float('inf')