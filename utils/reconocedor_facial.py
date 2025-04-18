import os
import face_recognition

def cargar_rostros(ruta_rostros='rostros_conocidos'):
    nombres = []
    codificaciones = []

    for archivo in os.listdir(ruta_rostros):
        ruta_completa = os.path.join(ruta_rostros, archivo)
        imagen = face_recognition.load_image_file(ruta_completa)
        codificacion = face_recognition.face_encodings(imagen)

        if codificacion:
            codificaciones.append(codificacion[0])
            nombre = os.path.splitext(archivo)[0]
            nombres.append(nombre)
        else:
            print(f"[ADVERTENCIA] No se pudo codificar: {archivo}")

    return nombres, codificaciones

def identificar_persona(rostro, codificaciones_conocidas, nombres_conocidos):
    codificacion_rostro = face_recognition.face_encodings(rostro)
    if not codificacion_rostro:
        return "Desconocido"

    resultados = face_recognition.compare_faces(codificaciones_conocidas, codificacion_rostro[0])
    if True in resultados:
        index = resultados.index(True)
        return nombres_conocidos[index]
    else:
        return "Desconocido"
