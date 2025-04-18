import cv2
import os
import time
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm
from utils.emocion import predecir_emocion
from datetime import datetime

# Ruta de rostros conocidos y carpeta de salida
RUTA_ROSTROS = "rostros/"
RUTA_SALIDAS = "salidas/"

# Crear carpeta de salidas si no existe
if not os.path.exists(RUTA_SALIDAS):
    os.makedirs(RUTA_SALIDAS)
    print(f"Carpeta '{RUTA_SALIDAS}' creada correctamente.")

# Lista para almacenar los resultados durante la ejecución
resultados_sesion = []

# Umbral de confianza para considerar una detección como válida (menor valor = más estricto)
UMBRAL_CONFIANZA = 9.0  # Bajamos aún más el umbral para ser mucho más estrictos

# Precargar rostros conocidos y embeddings
print("Cargando base de rostros...")
database = {}
for archivo in os.listdir(RUTA_ROSTROS):
    if archivo.lower().endswith((".jpg", ".png", ".jpeg")):
        nombre = os.path.splitext(archivo)[0]
        ruta = os.path.join(RUTA_ROSTROS, archivo)
        database[nombre] = ruta

print("Generando embeddings...")
embeddings_database = {}

for nombre, ruta in database.items():
    try:
        embedding_obj = DeepFace.represent(img_path=ruta, model_name="VGG-Face", enforce_detection=False)
        embedding = embedding_obj[0]["embedding"]
        embeddings_database[nombre] = embedding
        print(f"✓ Embedding generado para: {nombre}")
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo procesar el rostro de {nombre}: {e}")

print(f"Se cargaron {len(embeddings_database)} rostros registrados.")

# Inicializar cámara
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables de temporizador
ultimo_tiempo = 0
intervalo_segundos = 3
tiempo_inicio = datetime.now()

print("\n--- SISTEMA DE RECONOCIMIENTO FACIAL Y EMOCIONAL INICIADO ---\n")
print(f"Fecha y hora de inicio: {tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Umbral de confianza: {UMBRAL_CONFIANZA} (menor distancia = mayor confianza)")
print("Presiona 'q' para finalizar la ejecución\n")

# Bucle principal
while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    # Crear una copia del frame para dibujar
    frame_display = frame.copy()
    
    try:
        # Detectar todos los rostros en el frame
        resultados = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
        
        # Si se encontraron rostros, procesar cada uno
        if resultados:
            tiempo_actual = time.time()
            procesar_identidad = tiempo_actual - ultimo_tiempo >= intervalo_segundos
            
            # Lista para rastrear las ubicaciones de rostros y evitar múltiples detecciones del mismo
            rostros_detectados = []
            
            for i, rostro_data in enumerate(resultados):
                x = rostro_data["facial_area"]["x"]
                y = rostro_data["facial_area"]["y"]
                w = rostro_data["facial_area"]["w"]
                h = rostro_data["facial_area"]["h"]
                
                # Verificar si este rostro se solapa con alguno ya detectado
                es_duplicado = False
                for rx, ry, rw, rh in rostros_detectados:
                    # Calcular intersección
                    if (x < rx + rw and x + w > rx and 
                        y < ry + rh and y + h > ry):
                        area1 = w * h
                        area2 = rw * rh
                        overlap_area = max(0, min(x + w, rx + rw) - max(x, rx)) * max(0, min(y + h, ry + rh) - max(y, ry))
                        overlap_ratio = overlap_area / min(area1, area2)
                        if overlap_ratio > 0.5:  # Si se superponen más del 50%
                            es_duplicado = True
                            break
                
                if es_duplicado:
                    continue
                    
                # Registrar este rostro como detectado
                rostros_detectados.append((x, y, w, h))
                
                # Extraer el rostro
                rostro = frame[y:y+h, x:x+w]
                
                nombre = "..."
                emocion = "..."
                confianza = "N/A"
                es_conocido = False
                
                # Evaluar emoción en cada frame
                try:
                    emocion = predecir_emocion(rostro)
                except Exception as e:
                    emocion = "Error"
                    print(f"Error al predecir emoción: {e}")
                
                # Evaluar identidad cada ciertos segundos para no sobrecargar
                if procesar_identidad:
                    try:
                        embedding_obj = DeepFace.represent(img_path=rostro, model_name="VGG-Face", enforce_detection=False)
                        embedding_rostro = embedding_obj[0]["embedding"]
                        
                        nombre = "Desconocido"
                        menor_distancia = float('inf')
                        es_conocido = False
                        
                        # Versión mejorada del algoritmo de comparación
                        for persona, emb in embeddings_database.items():
                            distancia = norm(np.array(embedding_rostro) - np.array(emb))
                            
                            # Actualizar la menor distancia encontrada
                            if distancia < menor_distancia:
                                menor_distancia = distancia
                                
                                # Solo considerar como conocido si cumple con el umbral estricto
                                if distancia < UMBRAL_CONFIANZA:
                                    nombre = persona
                                    es_conocido = True
                                else:
                                    nombre = "Desconocido"
                                    es_conocido = False
                        
                        # Calcular nivel de confianza
                        if es_conocido:
                            confianza_valor = max(0, 100 - (menor_distancia * 10))
                            confianza = f"{confianza_valor:.1f}%"
                        else:
                            confianza = "N/A"
                            # Double check para asegurar que sea Desconocido
                            nombre = "Desconocido"
                        
                        # Registrar detección en consola con información adicional de confianza
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{timestamp}] Rostro #{i+1}: {nombre}" + 
                              f" - Emoción: {emocion}" + 
                              f" - Distancia: {menor_distancia:.2f}" + 
                              f" - Confianza: {confianza}" +
                              f" - {'Conocido' if es_conocido else 'No Registrado'}")
                        
                        # Solo guardar resultados de personas conocidas en el registro final
                        if es_conocido:
                            resultados_sesion.append({
                                "timestamp": timestamp,
                                "persona": nombre,
                                "emocion": emocion,
                                "confianza": confianza,
                                "es_conocido": True
                            })
                        else:
                            # Guardar desconocidos pero marcarlos explícitamente
                            resultados_sesion.append({
                                "timestamp": timestamp,
                                "persona": "Desconocido",
                                "emocion": emocion,
                                "confianza": "N/A",
                                "es_conocido": False
                            })
                        
                    except Exception as e:
                        print(f"Error al procesar identidad: {e}")
                        nombre = "Error"
                
                # Dibujar rectángulo y etiqueta
                # Color verde para conocidos, rojo para desconocidos
                color = (0, 255, 0) if es_conocido else (0, 0, 255)
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), color, 2)
                
                # Texto con nombre y emoción
                texto = f"{nombre} - {emocion}"
                
                # Añadir fondo oscuro para mejor visibilidad del texto
                texto_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame_display, (x, y - 25), (x + texto_size[0], y), color, -1)
                cv2.putText(frame_display, texto, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Actualizar el temporizador si procesamos identidades
            if procesar_identidad:
                ultimo_tiempo = tiempo_actual

    except Exception as e:
        print(f"Error general: {e}")
    
    # Mostrar indicador de personas detectadas
    num_personas = len(rostros_detectados)
    cv2.putText(frame_display, f"Personas detectadas: {num_personas}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostrar frame procesado
    cv2.imshow("Reconocimiento Facial y Emocional", frame_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Generar reporte al finalizar
tiempo_fin = datetime.now()
duracion = tiempo_fin - tiempo_inicio

# Crear nombre de archivo con fecha y hora actual
nombre_archivo = os.path.join(RUTA_SALIDAS, f"reporte_facial_{tiempo_fin.strftime('%Y%m%d_%H%M%S')}.txt")

with open(nombre_archivo, "w", encoding="utf-8") as archivo:
    archivo.write("=== REPORTE DE RECONOCIMIENTO FACIAL Y EMOCIONAL ===\n\n")
    archivo.write(f"Fecha de inicio: {tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}\n")
    archivo.write(f"Fecha de finalización: {tiempo_fin.strftime('%Y-%m-%d %H:%M:%S')}\n")
    archivo.write(f"Duración de la sesión: {duracion}\n")
    archivo.write(f"Total de detecciones: {len(resultados_sesion)}\n")
    archivo.write(f"Umbral de confianza: {UMBRAL_CONFIANZA}\n\n")
    
    # Separar resultados entre conocidos y desconocidos
    conocidos = [r for r in resultados_sesion if r["es_conocido"]]
    desconocidos = [r for r in resultados_sesion if not r["es_conocido"]]
    
    archivo.write(f"Personas conocidas detectadas: {len(conocidos)}\n")
    archivo.write(f"Personas desconocidas detectadas: {len(desconocidos)}\n\n")
    
    archivo.write("DETALLE DE DETECCIONES (PERSONAS REGISTRADAS):\n")
    archivo.write("-" * 70 + "\n")
    archivo.write(f"{'Fecha/Hora':<20} | {'Persona':<15} | {'Emoción':<12} | {'Confianza':<10}\n")
    archivo.write("-" * 70 + "\n")
    
    # Primero mostrar las personas conocidas
    for resultado in conocidos:
        archivo.write(f"{resultado['timestamp']:<20} | {resultado['persona']:<15} | {resultado['emocion']:<12} | {resultado['confianza']:<10}\n")
    
    # Sección separada para desconocidos
    archivo.write("\n\nDETALLE DE DETECCIONES (PERSONAS NO REGISTRADAS):\n")
    archivo.write("-" * 70 + "\n")
    archivo.write(f"{'Fecha/Hora':<20} | {'Estado':<15} | {'Emoción':<12}\n")
    archivo.write("-" * 70 + "\n")
    
    for resultado in desconocidos:
        archivo.write(f"{resultado['timestamp']:<20} | {'Desconocido':<15} | {resultado['emocion']:<12}\n")

    # Estadísticas adicionales solo para personas conocidas
    if conocidos:
        archivo.write("\n\nESTADÍSTICAS (SOLO PERSONAS REGISTRADAS):\n")
        archivo.write("-" * 50 + "\n")
        
        # Contar apariciones de personas conocidas
        personas = {}
        for res in conocidos:
            nombre = res['persona']
            if nombre in personas:
                personas[nombre] += 1
            else:
                personas[nombre] = 1
        
        archivo.write("Distribución de personas registradas detectadas:\n")
        for persona, cantidad in sorted(personas.items(), key=lambda x: x[1], reverse=True):
            porcentaje = (cantidad / len(conocidos)) * 100 if conocidos else 0
            archivo.write(f"  - {persona}: {cantidad} ({porcentaje:.1f}%)\n")
        
        # Contar emociones solo de personas conocidas
        emociones = {}
        for res in conocidos:
            emocion = res['emocion']
            if emocion in emociones:
                emociones[emocion] += 1
            else:
                emociones[emocion] = 1
        
        archivo.write("\nDistribución de emociones en personas registradas:\n")
        for emocion, cantidad in sorted(emociones.items(), key=lambda x: x[1], reverse=True):
            porcentaje = (cantidad / len(conocidos)) * 100 if conocidos else 0
            archivo.write(f"  - {emocion}: {cantidad} ({porcentaje:.1f}%)\n")

print(f"\n--- SISTEMA FINALIZADO ---")
print(f"Se ha generado el reporte: {nombre_archivo}")

cam.release()
cv2.destroyAllWindows()