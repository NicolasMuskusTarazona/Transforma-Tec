
PREGUNTAS = [
    {
        "id": 1,
        "tema": "Vectores",
        "pregunta": "¿Cuál es la norma del vector (3, 4)?",
        "respuesta": "5",
        "puntos": 10
    },
    {
        "id": 2,
        "tema": "Matrices",
        "pregunta": "¿Cuál es el determinante de [[2, 0], [0, 3]]?",
        "respuesta": "6",
        "puntos": 10
    },
    {
        "id": 3,
        "tema": "Sistemas",
        "pregunta": "¿Cuántas soluciones tiene un sistema compatible determinado?",
        "respuesta": "1",
        "puntos": 15
    },
    {
        "id": 4,
        "tema": "Transformaciones",
        "pregunta": "¿Qué transformación cambia el tamaño de una figura?",
        "respuesta": "escalamiento",
        "puntos": 15
    },
    {
        "id": 5,
        "tema": "Espacios vectoriales",
        "pregunta": "¿Cómo se llama un conjunto de vectores linealmente independientes que genera un espacio?",
        "respuesta": "base",
        "puntos": 20
    }
]


def obtener_preguntas():
    """Devuelve las preguntas disponibles."""
    return [pregunta.copy() for pregunta in PREGUNTAS]


def evaluar_respuesta(pregunta, respuesta):
    """Comprueba si una respuesta es correcta."""
    if not isinstance(respuesta, str):
        raise ValueError("La respuesta debe ser texto.")

    return respuesta.strip().lower() == pregunta["respuesta"].lower()


def calcular_puntaje(resultados):
    """Suma los puntos de las respuestas correctas."""
    return sum(
        resultado["puntos"]
        for resultado in resultados
        if resultado["correcta"]
    )


def determinar_nivel(puntaje):
    """Asigna un nivel según la puntuación."""
    if puntaje < 0:
        raise ValueError("El puntaje no puede ser negativo.")

    if puntaje < 20:
        return "Principiante"

    if puntaje < 40:
        return "Intermedio"

    if puntaje < 60:
        return "Avanzado"

    return "Experto"


def generar_retroalimentacion(correctas, total):
    """Genera un mensaje según el desempeño."""
    if total <= 0 or correctas < 0 or correctas > total:
        raise ValueError("Los resultados proporcionados no son válidos.")

    porcentaje = (correctas / total) * 100

    if porcentaje == 100:
        return "¡Excelente! Dominaste todos los retos."

    if porcentaje >= 60:
        return "¡Buen trabajo! Sigue fortaleciendo tus conocimientos."

    return "Sigue practicando. Cada reto es una oportunidad para mejorar."


def generar_resumen(resultados):
    """Genera el resumen de una partida."""
    if not resultados:
        raise ValueError("No hay resultados para generar el resumen.")

    correctas = sum(
        1 for resultado in resultados if resultado["correcta"]
    )

    total = len(resultados)
    puntaje = calcular_puntaje(resultados)

    return {
        "correctas": correctas,
        "incorrectas": total - correctas,
        "total": total,
        "puntaje": puntaje,
        "nivel": determinar_nivel(puntaje),
        "retroalimentacion": generar_retroalimentacion(correctas, total)
    }
