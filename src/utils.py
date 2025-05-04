import os
from datetime import datetime

def generate_report(resultados_sesion, output_dir, tiempo_inicio, tiempo_fin, umbral_confianza=9.0):
    """Genera un reporte detallado de las detecciones."""
    os.makedirs(output_dir, exist_ok=True)
    nombre_archivo = os.path.join(output_dir, f"reporte_{tiempo_fin.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.write("=== REPORTE DE RECONOCIMIENTO FACIAL Y EMOCIONAL ===\n\n")
        f.write(f"Fecha de inicio: {tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fecha de finalización: {tiempo_fin.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duración: {tiempo_fin - tiempo_inicio}\n")
        f.write(f"Total detecciones: {len(resultados_sesion)}\n")
        f.write(f"Umbral de confianza: {umbral_confianza}\n\n")
        
        conocidos = [r for r in resultados_sesion if r["es_conocido"]]
        desconocidos = [r for r in resultados_sesion if not r["es_conocido"]]
        
        f.write(f"Personas conocidas: {len(conocidos)}\n")
        f.write(f"Personas desconocidas: {len(desconocidos)}\n\n")
        
        f.write("DETALLE CONOCIDOS:\n")
        f.write(f"{'Fecha/Hora':<20} | {'Persona':<15} | {'Emoción':<12} | {'Confianza':<10}\n")
        f.write("-" * 70 + "\n")
        for r in conocidos:
            f.write(f"{r['timestamp']:<20} | {r['persona']:<15} | {r['emocion']:<12} | {r['confianza']:<10}\n")
        
        f.write("\nDETALLE DESCONOCIDOS:\n")
        f.write(f"{'Fecha/Hora':<20} | {'Estado':<15} | {'Emoción':<12}\n")
        f.write("-" * 70 + "\n")
        for r in desconocidos:
            f.write(f"{r['timestamp']:<20} | {'Desconocido':<15} | {r['emocion']:<12}\n")
        
        # Estadísticas
        if conocidos:
            f.write("\nESTADÍSTICAS (PERSONAS CONOCIDAS):\n")
            f.write("-" * 50 + "\n")
            
            personas = {}
            for r in conocidos:
                nombre = r['persona']
                personas[nombre] = personas.get(nombre, 0) + 1
            
            f.write("Distribución de personas:\n")
            for persona, cantidad in sorted(personas.items(), key=lambda x: x[1], reverse=True):
                porcentaje = (cantidad / len(conocidos)) * 100
                f.write(f"  - {persona}: {cantidad} ({porcentaje:.1f}%)\n")
            
            emociones = {}
            for r in conocidos:
                emocion = r['emocion']
                emociones[emocion] = emociones.get(emocion, 0) + 1
            
            f.write("\nDistribución de emociones:\n")
            for emocion, cantidad in sorted(emociones.items(), key=lambda x: x[1], reverse=True):
                porcentaje = (cantidad / len(conocidos)) * 100
                f.write(f"  - {emocion}: {cantidad} ({porcentaje:.1f}%)\n")
    
    print(f"Reporte generado: {nombre_archivo}")
    return nombre_archivo