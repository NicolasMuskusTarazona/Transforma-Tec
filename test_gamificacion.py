from src.gamificacion import (
    generar_retos, evaluar_reto, PartidaGamificada, PUNTOS_DESBLOQUEO
)

# La semilla garantiza que la selección y los valores de los retos sean reproducibles.
retos_a = generar_retos(seed=2026)
retos_b = generar_retos(seed=2026)
assert retos_a == retos_b
assert len(retos_a) == 5
assert all(len(nivel["retos"]) == 3 for nivel in retos_a)
assert sum(len(n["retos"]) for n in retos_a) == 15

# Evaluación numérica y textual.
assert evaluar_reto({"tipo": "numero", "respuesta": 5, "tolerancia": 1e-6}, "5")
assert evaluar_reto({"tipo": "texto", "respuesta": "Sí"}, "si")
assert not evaluar_reto({"tipo": "numero", "respuesta": 5, "tolerancia": 1e-6}, "7")

# Un recorrido perfecto completa los cinco niveles y otorga cinco insignias.
partida = PartidaGamificada(seed=2026)
while not partida.finalizada:
    reto = partida.niveles[partida.nivel_actual]["retos"][partida.reto_actual]
    resultado = partida.responder(str(reto["respuesta"]))
    assert resultado["correcta"]

resumen = partida.resumen()
assert resumen["estado"] == "Completada"
assert resumen["correctas"] == 15
assert resumen["incorrectas"] == 0
assert len(resumen["insignias"]) == 5
assert resumen["puntaje"] == 150

# Tres errores agotan las vidas e impiden desbloquear el nivel siguiente.
partida_error = PartidaGamificada(seed=2026)
for _ in range(3):
    partida_error.responder("__respuesta_incorrecta__")
assert partida_error.finalizada
assert partida_error.vidas == 0
assert len(partida_error.insignias) == 0
assert partida_error.resumen()["estado"] == "Finalizada"

print("Todas las pruebas del sistema de gamificación fueron exitosas.")
print("15 retos reproducibles, evaluación, vidas, puntaje, insignias y desbloqueo verificados.")
