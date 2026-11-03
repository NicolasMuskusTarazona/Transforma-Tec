
from src.gamificacion import (
    obtener_preguntas,
    evaluar_respuesta,
    calcular_puntaje,
    determinar_nivel,
    generar_retroalimentacion,
    generar_resumen
)

# Obtener preguntas
preguntas = obtener_preguntas()

assert len(preguntas) == 5

# Evaluar respuestas
assert evaluar_respuesta(preguntas[0], "5")
assert not evaluar_respuesta(preguntas[0], "10")

# Calcular puntuación
resultados = [
    {"correcta": True, "puntos": 10},
    {"correcta": True, "puntos": 15},
    {"correcta": False, "puntos": 20}
]

assert calcular_puntaje(resultados) == 25

# Comprobar niveles
assert determinar_nivel(10) == "Principiante"
assert determinar_nivel(30) == "Intermedio"
assert determinar_nivel(50) == "Avanzado"
assert determinar_nivel(70) == "Experto"

# Retroalimentación
assert "Excelente" in generar_retroalimentacion(5, 5)

# Resumen de partida
resumen = generar_resumen(resultados)

assert resumen["correctas"] == 2
assert resumen["incorrectas"] == 1
assert resumen["puntaje"] == 25
assert resumen["nivel"] == "Intermedio"

print("Todas las pruebas de gamificación fueron exitosas.")
