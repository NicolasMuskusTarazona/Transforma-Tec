"""Sistema de gamificación de Transforma-Tec.

Genera tres retos reproducibles por nivel, evalúa respuestas, administra vidas,
puntajes, insignias y desbloqueo progresivo. La interfaz interactiva utiliza
ipywidgets en Google Colab.
"""
from __future__ import annotations

import math
import random
import unicodedata
from typing import Any, Callable


NIVELES = [
    {"numero": 1, "nombre": "Vectores", "insignia": "Explorador Vectorial"},
    {"numero": 2, "nombre": "Matrices", "insignia": "Arquitecto de Matrices"},
    {"numero": 3, "nombre": "Sistemas de ecuaciones", "insignia": "Maestro de Gauss"},
    {"numero": 4, "nombre": "Transformaciones 2D", "insignia": "Rotador Experto"},
    {"numero": 5, "nombre": "Transformaciones compuestas", "insignia": "Arquitecto Homogéneo"},
]
PUNTOS_RETO = 10
PUNTOS_DESBLOQUEO = 20
VIDAS_INICIALES = 3


def _normalizar_texto(valor: Any) -> str:
    """Normaliza texto ignorando mayúsculas, tildes y espacios externos."""
    texto = str(valor).strip().lower()
    return "".join(
        caracter for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )


def _reto(pregunta: str, respuesta: Any, tipo: str = "numero",
          tolerancia: float = 1e-6, explicacion: str = "") -> dict:
    return {
        "pregunta": pregunta,
        "respuesta": respuesta,
        "tipo": tipo,
        "tolerancia": tolerancia,
        "explicacion": explicacion,
        "puntos": PUNTOS_RETO,
    }


def generar_retos(seed: int = 2026) -> list[dict]:
    """Genera 15 retos (3 por nivel) de forma reproducible con una semilla."""
    rng = random.Random(seed)
    retos_por_nivel = []

    # Nivel 1: Vectores
    banco = []
    for _ in range(12):
        a, b = rng.randint(1, 8), rng.randint(1, 8)
        banco.extend([
            _reto(f"Calcula la norma del vector ({a}, {b}).",
                  math.sqrt(a*a+b*b), explicacion="La norma es sqrt(x^2 + y^2)."),
            _reto(f"Calcula el producto punto de ({a}, {b}) y ({b}, {a}).",
                  2*a*b, explicacion="Multiplica las componentes correspondientes y suma."),
            _reto(f"Si u=({a}, {b}) y v=({b}, {a}), ¿cuál es la primera componente de u+v?",
                  a+b, explicacion="La suma se realiza componente a componente."),
        ])
    retos_por_nivel.append(rng.sample(banco, 3))

    # Nivel 2: Matrices
    banco = []
    for _ in range(12):
        a, b, c, d = [rng.randint(1, 6) for _ in range(4)]
        banco.extend([
            _reto(f"Calcula el determinante de [[{a}, {b}], [{c}, {d}]].",
                  a*d-b*c, explicacion="Para 2x2: det(A)=a*d-b*c."),
            _reto(f"¿Cuál es la traza de la matriz [[{a}, {b}], [{c}, {d}]]?",
                  a+d, explicacion="La traza es la suma de los elementos diagonales."),
            _reto(f"¿Cuánto vale el elemento (1,2) de [[{a}, {b}], [{c}, {d}]]?",
                  b, explicacion="El elemento de la fila 1, columna 2 es el segundo valor de la primera fila."),
        ])
    retos_por_nivel.append(rng.sample(banco, 3))

    # Nivel 3: Sistemas
    banco = []
    for _ in range(12):
        x, y = rng.randint(-5, 8), rng.randint(-5, 8)
        a, b = rng.randint(1, 5), rng.randint(1, 5)
        c, d = rng.randint(1, 5), rng.randint(1, 5)
        while a*d-b*c == 0:
            d = rng.randint(1, 5)
        banco.extend([
            _reto(f"En el sistema {a}x + {b}y = {a*x+b*y}; {c}x + {d}y = {c*x+d*y}, ¿cuánto vale x?",
                  x, explicacion="La solución se obtiene resolviendo simultáneamente ambas ecuaciones."),
            _reto(f"¿Cuántas soluciones tiene un sistema compatible determinado?",
                  1, explicacion="Un sistema compatible determinado tiene una única solución."),
            _reto(f"¿Qué tipo de sistema tiene infinitas soluciones? (responde: compatible indeterminado, incompatible o compatible determinado)",
                  "compatible indeterminado", "texto",
                  explicacion="Tiene infinitas soluciones y sus ecuaciones son dependientes."),
        ])
    retos_por_nivel.append(rng.sample(banco, 3))

    # Nivel 4: Transformaciones 2D
    banco = []
    for _ in range(12):
        x, y, k = rng.randint(1, 6), rng.randint(1, 6), rng.randint(2, 4)
        banco.extend([
            _reto(f"Un punto ({x}, {y}) se escala uniformemente por k={k}. ¿Cuál es su nueva coordenada x?",
                  k*x, explicacion="El escalamiento multiplica cada coordenada por k."),
            _reto(f"Se traslada el punto ({x}, {y}) mediante el vector ({k}, {-k}). ¿Cuál es su nueva coordenada y?",
                  y-k, explicacion="La traslación suma el vector de desplazamiento al punto."),
            _reto(f"¿Cuál es el determinante de una rotación 2D pura?",
                  1, explicacion="Las matrices de rotación conservan orientación y área; su determinante es 1."),
        ])
    retos_por_nivel.append(rng.sample(banco, 3))

    # Nivel 5: Transformaciones compuestas y homogéneas
    banco = []
    for _ in range(12):
        x, y, tx, ty, k = (rng.randint(1, 5), rng.randint(1, 5),
                          rng.randint(-4, 4), rng.randint(-4, 4), rng.randint(2, 3))
        banco.extend([
            _reto(f"En coordenadas homogéneas, traslada ({x}, {y}) con ({tx}, {ty}). ¿Cuál es la nueva coordenada x?",
                  x+tx, explicacion="La matriz homogénea de traslación suma tx a la coordenada x."),
            _reto(f"Primero escala por {k} y luego traslada en x por {tx}. Para el punto ({x}, {y}), ¿cuál es la coordenada x final?",
                  k*x+tx, explicacion="Se aplica primero el escalamiento y después la traslación."),
            _reto("¿El orden de composición de dos transformaciones lineales puede cambiar el resultado? (responde: si o no)",
                  "si", "texto", explicacion="En general, la composición no es conmutativa: AB puede ser distinto de BA."),
        ])
    retos_por_nivel.append(rng.sample(banco, 3))

    return [
        {"nivel": NIVELES[i].copy(), "retos": retos_por_nivel[i]}
        for i in range(5)
    ]


def evaluar_reto(reto: dict, respuesta: Any) -> bool:
    """Evalúa un reto numérico o textual con validación de entrada."""
    if respuesta is None or str(respuesta).strip() == "":
        return False
    if reto["tipo"] == "texto":
        return _normalizar_texto(respuesta) == _normalizar_texto(reto["respuesta"])
    try:
        valor = float(str(respuesta).replace(",", "."))
        return math.isfinite(valor) and math.isclose(
            valor, float(reto["respuesta"]),
            rel_tol=1e-6, abs_tol=reto["tolerancia"]
        )
    except (TypeError, ValueError):
        return False


class PartidaGamificada:
    """Administra el estado y las reglas de una partida de cinco niveles."""

    def __init__(self, seed: int = 2026, vidas: int = VIDAS_INICIALES):
        if vidas < 1:
            raise ValueError("La partida debe iniciar con al menos una vida.")
        self.seed = seed
        self.niveles = generar_retos(seed)
        self.vidas = vidas
        self.puntaje = 0
        self.nivel_actual = 0
        self.reto_actual = 0
        self.resultados = []
        self.insignias = []
        self.finalizada = False
        self.nivel_superado = False

    def responder(self, respuesta: Any) -> dict:
        """Registra una respuesta y actualiza vidas, puntos y desbloqueos."""
        if self.finalizada:
            raise RuntimeError("La partida ya finalizó.")
        nivel = self.niveles[self.nivel_actual]
        reto = nivel["retos"][self.reto_actual]
        correcta = evaluar_reto(reto, respuesta)
        if correcta:
            self.puntaje += reto["puntos"]
        else:
            self.vidas -= 1

        registro = {
            "nivel": nivel["nivel"]["numero"],
            "tema": nivel["nivel"]["nombre"],
            "reto": self.reto_actual + 1,
            "correcta": correcta,
            "respuesta_usuario": str(respuesta),
            "respuesta_correcta": str(reto["respuesta"]),
            "puntos": reto["puntos"] if correcta else 0,
            "vidas_restantes": self.vidas,
        }
        self.resultados.append(registro)
        self.reto_actual += 1

        mensaje = "¡Correcto!" if correcta else f"Respuesta esperada: {reto['respuesta']}."
        mensaje += " " + reto["explicacion"]

        if self.vidas <= 0:
            self.finalizada = True
            mensaje += " Se agotaron las vidas."
        elif self.reto_actual == 3:
            aciertos = sum(
                r["correcta"] for r in self.resultados
                if r["nivel"] == nivel["nivel"]["numero"]
            )
            if aciertos >= 2:
                self.insignias.append(nivel["nivel"]["insignia"])
                self.nivel_superado = True
                if self.nivel_actual == 4:
                    self.finalizada = True
                    mensaje += " ¡Completaste los cinco niveles!"
                else:
                    self.nivel_actual += 1
                    self.reto_actual = 0
                    mensaje += f" Nivel superado. Desbloqueaste: {self.niveles[self.nivel_actual]['nivel']['nombre']}."
            else:
                self.finalizada = True
                mensaje += " No alcanzaste los 20 puntos requeridos para desbloquear el siguiente nivel."
        return {
            "correcta": correcta,
            "mensaje": mensaje,
            "puntaje": self.puntaje,
            "vidas": self.vidas,
            "finalizada": self.finalizada,
            "nivel_actual": self.nivel_actual + 1,
            "reto_siguiente": self.reto_actual + 1,
        }

    def tabla_progreso(self) -> list[dict]:
        """Devuelve el progreso de los cinco niveles para mostrarlo en tabla."""
        filas = []
        for item in self.niveles:
            numero = item["nivel"]["numero"]
            registros = [r for r in self.resultados if r["nivel"] == numero]
            correctas = sum(r["correcta"] for r in registros)
            desbloqueado = numero == 1 or numero <= self.nivel_actual + 1
            superado = any(i == item["nivel"]["insignia"] for i in self.insignias)
            filas.append({
                "Nivel": numero,
                "Tema": item["nivel"]["nombre"],
                "Retos respondidos": len(registros),
                "Correctas": correctas,
                "Puntos": sum(r["puntos"] for r in registros),
                "Estado": (
                    "Superado" if superado else
                    ("En curso" if desbloqueado and numero == self.nivel_actual + 1 and not self.finalizada else
                     ("Disponible" if desbloqueado and not self.finalizada else
                      ("No superado" if registros else "Bloqueado")))
                ),
                "Insignia": item["nivel"]["insignia"] if superado else "Pendiente",
            })
        return filas

    def resumen(self) -> dict:
        """Resume los resultados de la partida."""
        correctas = sum(r["correcta"] for r in self.resultados)
        return {
            "puntaje": self.puntaje,
            "vidas_restantes": self.vidas,
            "retos_respondidos": len(self.resultados),
            "correctas": correctas,
            "incorrectas": len(self.resultados) - correctas,
            "insignias": self.insignias.copy(),
            "estado": "Completada" if self.finalizada and self.nivel_actual == 4 and len(self.insignias) == 5
                     else ("Finalizada" if self.finalizada else "En curso"),
        }


def iniciar_juego(seed: int = 2026):
    """Inicia la demo interactiva en Google Colab mediante ipywidgets."""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, HTML
    except ImportError as exc:
        raise ImportError("Instala ipywidgets para ejecutar la demo interactiva.") from exc

    partida = PartidaGamificada(seed=seed)
    titulo = widgets.HTML("<h2>🎮 Transforma-Tec: El Gran Reto Algebraico</h2>"
                          "<p>Supera cada nivel con al menos 2 respuestas correctas de 3. "
                          "Tienes 3 vidas para completar el desafío.</p>")
    estado = widgets.HTML()
    pregunta = widgets.HTML()
    respuesta = widgets.Text(placeholder="Escribe tu respuesta...")
    boton = widgets.Button(description="Enviar respuesta", button_style="success",
                           icon="check")
    salida = widgets.Output()
    progreso = widgets.Output()
    display(titulo, estado, pregunta, respuesta, boton, salida, progreso)

    def actualizar():
        estado.value = (
            f"<b>Puntaje:</b> {partida.puntaje} &nbsp; "
            f"<b>Vidas:</b> {'❤️ ' * partida.vidas} &nbsp; "
            f"<b>Insignias:</b> {len(partida.insignias)}/5"
        )
        with progreso:
            clear_output()
            try:
                import pandas as pd
                display(pd.DataFrame(partida.tabla_progreso()))
            except ImportError:
                for fila in partida.tabla_progreso():
                    print(fila)
        if partida.finalizada:
            boton.disabled = True
            respuesta.disabled = True
            pregunta.value = "<h3>🏁 Partida finalizada</h3>"
            with salida:
                clear_output()
                print(partida.resumen())
                print("\nInsignias obtenidas:", ", ".join(partida.insignias) or "Ninguna")
            return
        nivel = partida.niveles[partida.nivel_actual]
        reto = nivel["retos"][partida.reto_actual]
        pregunta.value = (
            f"<h3>Nivel {nivel['nivel']['numero']}: {nivel['nivel']['nombre']}</h3>"
            f"<p><b>Reto {partida.reto_actual + 1} de 3</b> · "
            f"{reto['puntos']} puntos</p><p>{reto['pregunta']}</p>"
        )

    def enviar(_):
        if not respuesta.value.strip():
            with salida:
                clear_output()
                print("Escribe una respuesta antes de continuar.")
            return
        reto = partida.niveles[partida.nivel_actual]["retos"][partida.reto_actual]
        resultado = partida.responder(respuesta.value)
        with salida:
            clear_output()
            print(resultado["mensaje"])
            print(f"Puntaje acumulado: {resultado['puntaje']} | Vidas: {resultado['vidas']}")
            if resultado["correcta"]:
                print(f"Respuesta validada: {reto['respuesta']}")
        respuesta.value = ""
        actualizar()

    boton.on_click(enviar)
    actualizar()
    return partida


# Compatibilidad con funciones sencillas de versiones anteriores.
def obtener_preguntas(seed: int = 2026) -> list[dict]:
    """Devuelve los 15 retos agrupados por nivel."""
    return generar_retos(seed)


def evaluar_respuesta(pregunta: dict, respuesta: Any) -> bool:
    """Alias compatible para evaluar_reto."""
    if "tipo" not in pregunta:
        pregunta = {**pregunta, "tipo": "texto", "tolerancia": 1e-6}
    return evaluar_reto(pregunta, respuesta)
