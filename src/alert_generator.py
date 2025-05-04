from collections import Counter

class AlertGenerator:
    def __init__(self):
        # Definir las combinaciones de emociones que generan alertas
        self.alert_rules = [
            {
                "emotions": {"angry", "disgust"},
                "description": "La combinación puede indicar una fuerte aversión o frustración.",
                "solution": "Investigar la situación específica que genera ambas emociones. Ofrecer un espacio seguro para que el estudiante exprese sus sentimientos y necesidades. Buscar soluciones concretas a la situación desencadenante."
            },
            {
                "emotions": {"angry", "neutral"},
                "description": "El enojo podría estar internalizado o el estudiante podría estar reprimiendo sus sentimientos.",
                "solution": "Ofrecer espacios de diálogo individual para explorar sus emociones. Estar atento a posibles señales de malestar."
            },
            {
                "emotions": {"angry", "sad"},
                "description": "Esta combinación puede ser preocupante e indicar frustración profunda o desesperanza.",
                "solution": "Ofrecer apoyo psicológico de inmediato. Investigar la causa subyacente de ambas emociones y brindar un acompañamiento integral."
            },
            {
                "emotions": {"disgust", "sad"},
                "description": "Esta combinación puede indicar una profunda aversión hacia una situación que también genera tristeza o desesperanza.",
                "solution": "Ofrecer apoyo emocional y buscar soluciones a la situación que genera estas emociones. Considerar apoyo psicológico."
            }
        ]

    def generate_alerts(self, conocidos):
        """Genera alertas basadas en las emociones de los estudiantes."""
        alerts = []
        
        # Agrupar emociones por estudiante
        emociones_por_estudiante = {}
        for deteccion in conocidos:
            persona = deteccion["persona"]
            emocion = deteccion["emocion"]
            if persona not in emociones_por_estudiante:
                emociones_por_estudiante[persona] = []
            emociones_por_estudiante[persona].append(emocion)
        
        # Analizar cada estudiante
        for persona, emociones in emociones_por_estudiante.items():
            # Obtener emociones únicas
            emociones_unicas = set(emociones)
            # Verificar cada regla de alerta
            for rule in self.alert_rules:
                if rule["emotions"].issubset(emociones_unicas):
                    alerts.append({
                        "persona": persona,
                        "emotions": ", ".join(rule["emotions"]),
                        "description": rule["description"],
                        "solution": rule["solution"]
                    })
        
        return alerts