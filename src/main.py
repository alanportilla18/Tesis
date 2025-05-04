import cv2
import time
import os
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from src.face_recognizer import FaceRecognizer
from src.emotion_detector import EmotionDetector
from src.alert_generator import AlertGenerator

# Configuración
UMBRAL_CONFIANZA = 12.0
INTERVALO_PROCESAMIENTO = 15  # Intervalo en segundos
OUTPUT_DIR = "outputs"

def ensure_output_dir():
    """Crea la carpeta outputs si no existe."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Carpeta {OUTPUT_DIR} creada.")

def format_timedelta(td):
    """Formatea una diferencia de tiempo como HH:MM:SS.mmmmmm."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    microseconds = td.microseconds
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}"

def save_report(conocidos, desconocidos, start_time, end_time, umbral_confianza):
    """Guarda el reporte en un archivo de texto con el formato especificado."""
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"reporte_{timestamp}.txt")
    
    duration = end_time - start_time
    total_detecciones = len(conocidos) + len(desconocidos)
    
    # Calcular estadísticas
    personas_counter = Counter([c["persona"] for c in conocidos])
    emociones_counter = Counter([c["emocion"] for c in conocidos])
    total_conocidos = len(conocidos)
    
    # Generar alertas
    alert_generator = AlertGenerator()
    alerts = alert_generator.generate_alerts(conocidos)
    
    with open(report_path, "w", encoding="utf-8") as f:
        # Encabezado
        f.write("=== REPORTE DE RECONOCIMIENTO FACIAL Y EMOCIONAL ===\n\n")
        f.write(f"Fecha de inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fecha de finalización: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duración: {format_timedelta(duration)}\n")
        f.write(f"Total detecciones: {total_detecciones}\n")
        f.write(f"Umbral de confianza: {umbral_confianza}\n\n")
        f.write(f"Personas conocidas: {len(conocidos)}\n")
        f.write(f"Personas desconocidas: {len(desconocidos)}\n\n")
        
        # Detalle conocidos
        f.write("DETALLE CONOCIDOS:\n")
        f.write("Fecha/Hora           | Persona         | Emoción      | Confianza \n")
        f.write("-" * 66 + "\n")
        for c in conocidos:
            f.write(f"{c['timestamp']}  | {c['persona']:<15} | {c['emocion']:<12} | {c['confianza']}\n")
        
        # Detalle desconocidos
        f.write("\nDETALLE DESCONOCIDOS:\n")
        f.write("Fecha/Hora           | Estado          | Emoción     \n")
        f.write("-" * 66 + "\n")
        for d in desconocidos:
            f.write(f"{d['timestamp']}  | {d['estado']:<15} | {d['emocion']:<12}\n")
        
        # Estadísticas
        f.write("\nESTADÍSTICAS (PERSONAS CONOCIDAS):\n")
        f.write("-" * 50 + "\n")
        f.write("Distribución de personas:\n")
        for persona, count in personas_counter.items():
            porcentaje = (count / total_conocidos * 100) if total_conocidos > 0 else 0
            f.write(f"  - {persona}: {count} ({porcentaje:.1f}%)\n")
        
        f.write("\nDistribución de emociones:\n")
        for emocion, count in emociones_counter.items():
            porcentaje = (count / total_conocidos * 100) if total_conocidos > 0 else 0
            f.write(f"  - {emocion}: {count} ({porcentaje:.1f}%)\n")
        
        # Alertas
        f.write("\nALERTAS:\n")
        f.write("-" * 50 + "\n")
        if alerts:
            for alert in alerts:
                f.write(f"Estudiante: {alert['persona']}\n")
                f.write(f"Emociones detectadas: {alert['emotions']}\n")
                f.write(f"Descripción: {alert['description']}\n")
                f.write(f"Solución propuesta: {alert['solution']}\n")
                f.write("-" * 50 + "\n")
        else:
            f.write("No se generaron alertas.\n")
    
    print(f"Reporte guardado en: {report_path}")

def main():
    # Crear carpeta de salida
    ensure_output_dir()

    # Inicializar reconocedor de rostros y detector de emociones
    face_recognizer = FaceRecognizer(umbral_confianza=UMBRAL_CONFIANZA)
    emotion_detector = EmotionDetector(model_type="cnn")  #cambiar entre modelos con las variabels "cnn" y "resnet"

    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    # Variables para el control del intervalo y resultados
    ultimo_procesamiento = time.time()
    start_time = datetime.now()
    conocidos = []
    desconocidos = []

    print("Iniciando captura de video. Procesando cada {} segundos...".format(INTERVALO_PROCESAMIENTO))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break

        # Mostrar el frame en tiempo real (sin procesar)
        cv2.imshow("Reconocimiento Facial y Emociones", frame)

        # Verificar si ha pasado el intervalo de procesamiento
        tiempo_actual = time.time()
        if tiempo_actual - ultimo_procesamiento >= INTERVALO_PROCESAMIENTO:
            print(f"\nProcesando frame a los {tiempo_actual:.2f} segundos...")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Detectar rostros
            faces = face_recognizer.detect_faces(frame)
            for face_info in faces:
                x, y, w, h = (
                    face_info["facial_area"]["x"],
                    face_info["facial_area"]["y"],
                    face_info["facial_area"]["w"],
                    face_info["facial_area"]["h"],
                )
                # Asegurarse de que las coordenadas estén dentro del frame
                if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                    rostro = frame[y:y+h, x:x+w]
                    # Reconocer rostro
                    nombre, confianza, distancia = face_recognizer.recognize_face(rostro)
                    # Detectar emoción
                    emocion = emotion_detector.predict_emotion(rostro)

                    # Almacenar resultados
                    if nombre == "Desconocido":
                        desconocidos.append({
                            "timestamp": timestamp,
                            "estado": "Desconocido",
                            "emocion": emocion
                        })
                    else:
                        conocidos.append({
                            "timestamp": timestamp,
                            "persona": nombre,
                            "emocion": emocion,
                            "confianza": confianza
                        })

                    # Mostrar resultados
                    print(f"Persona: {nombre}, Confianza: {confianza}, Emoción: {emocion}")
                    # Dibujar rectángulo y etiquetas
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    etiqueta = f"{nombre} ({confianza}) - {emocion}"
                    cv2.putText(
                        frame, etiqueta, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2
                    )
                else:
                    print("Advertencia: Coordenadas del rostro fuera de los límites del frame")

            # Actualizar el tiempo del último procesamiento
            ultimo_procesamiento = tiempo_actual

        # Mostrar el frame (con anotaciones si se procesó)
        cv2.imshow("Reconocimiento Facial y Emociones", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Guardar reporte al finalizar
    end_time = datetime.now()
    save_report(conocidos, desconocidos, start_time, end_time, UMBRAL_CONFIANZA)

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Programa terminado.")

if __name__ == "__main__":
    main()