import numpy as np

from src.transformaciones import (
    aplicar_transformacion,
    matriz_rotacion,
    matriz_escalamiento,
    matriz_reflexion,
    matriz_cizallamiento,
    componer_transformaciones,
    es_transformacion_lineal,
    matriz_homogenea,
    matriz_traslacion,
    rotacion_homogenea,
    escalamiento_homogeneo,
    reflexion_homogenea,
    cizallamiento_homogeneo,
    rotacion_alrededor_de_punto,
    aplicar_homogenea,
    crear_triangulo,
    crear_cuadrado,
    crear_sprite,
    area_poligono,
    longitudes_lados,
    analizar_transformacion,
    verificar_orden_importa,
    graficar_poligono_transformado
)

# Rotación de 90 grados
rotacion = matriz_rotacion(90)
resultado = aplicar_transformacion(rotacion, [1, 0])

assert np.allclose(resultado, [0, 1], atol=1e-10)

# Escalamiento
escala = matriz_escalamiento(2, 3)
assert np.allclose(
    aplicar_transformacion(escala, [2, 2]),
    [4, 6]
)

# Reflexión respecto al eje X
reflexion = matriz_reflexion("x")
assert np.allclose(
    aplicar_transformacion(reflexion, [3, 4]),
    [3, -4]
)

# Cizallamiento
cizalla = matriz_cizallamiento(kx=2)
assert np.allclose(
    aplicar_transformacion(cizalla, [1, 1]),
    [3, 1]
)

# Composición: primero escala y después rotación
compuesta = componer_transformaciones(
    matriz_escalamiento(2, 2),
    matriz_rotacion(90)
)

assert np.allclose(
    aplicar_transformacion(compuesta, [1, 0]),
    [0, 2],
    atol=1e-10
)

assert es_transformacion_lineal(rotacion)

# ----------------------------------------------------------------------
# Coordenadas homogéneas y polígonos
# ----------------------------------------------------------------------
triangulo = crear_triangulo()
cuadrado = crear_cuadrado()
sprite = crear_sprite()

# Áreas conocidas
assert np.isclose(area_poligono(triangulo), 6)
assert np.isclose(area_poligono(cuadrado), 4)
assert np.isclose(area_poligono(sprite), 10)  # rectángulo 3x2 (6) + punta triangular (4)
assert np.allclose(longitudes_lados(cuadrado), [2, 2, 2, 2])

# Traslación: (x, y) -> (x + 2, y + 1)
M = matriz_traslacion(2, 1)
assert np.allclose(M, [[1, 0, 2], [0, 1, 1], [0, 0, 1]])
assert np.allclose(aplicar_homogenea(M, triangulo), [[3, 2], [7, 2], [5, 5]])

# La traslación no es lineal: no fija el origen en 2D, pero en 3x3 sí es matriz
assert np.isclose(np.linalg.det(M), 1)

# Rotación de 90° del triángulo: A'(-1, 1), B'(-1, 5), C'(-4, 3)
res = analizar_transformacion(rotacion_homogenea(90), triangulo)
assert np.allclose(res["transformado"], [[-1, 1], [-1, 5], [-4, 3]], atol=1e-10)
assert res["conserva_area"] and res["conserva_longitudes"]
assert np.isclose(res["determinante"], 1)

# Escalamiento k = 2: área x4, longitudes x2, det = 4
res = analizar_transformacion(escalamiento_homogeneo(2, 2), triangulo)
assert np.allclose(res["transformado"], [[2, 2], [10, 2], [6, 8]])
assert np.isclose(res["area_transformada"], 24)
assert np.isclose(res["razon_area"], 4) and np.isclose(res["determinante"], 4)
assert not res["conserva_area"] and not res["conserva_longitudes"]

# Traslación conserva área y longitudes
res = analizar_transformacion(matriz_traslacion(2, 1), cuadrado)
assert res["conserva_area"] and res["conserva_longitudes"]

# Reflexión: det = -1, conserva área y longitudes
res = analizar_transformacion(reflexion_homogenea("x"), sprite)
assert np.isclose(res["determinante"], -1)
assert res["conserva_area"] and res["conserva_longitudes"]

# Cizallamiento: det = 1, conserva área pero no longitudes
res = analizar_transformacion(cizallamiento_homogeneo(kx=1), cuadrado)
assert np.isclose(res["determinante"], 1)
assert res["conserva_area"] and not res["conserva_longitudes"]

# Escalamiento no uniforme: razón de área = sx * sy
res = analizar_transformacion(escalamiento_homogeneo(2, 3), cuadrado)
assert np.isclose(res["razon_area"], 6)

# Rotación alrededor de un punto: el centro permanece fijo
R = rotacion_alrededor_de_punto(90, (3, 3))
assert np.allclose(aplicar_homogenea(R, [[3, 3], [4, 3], [3, 4]]),
                   [[3, 3], [3, 4], [2, 3]], atol=1e-10)

# Composición (el último de la lista se aplica al final): rotar 45° y escalar x2
comp = componer_transformaciones(rotacion_homogenea(45), escalamiento_homogeneo(2, 2))
assert np.allclose(comp[:2, :2], 2 * matriz_rotacion(45))
assert np.isclose(np.linalg.det(comp[:2, :2]), 4)

# El orden importa: traslación y rotación no conmutan (AB != BA)
T = matriz_traslacion(2, 1)
Rz = rotacion_homogenea(90)
AB, BA, distintas = verificar_orden_importa(T, Rz)
assert distintas
assert not np.allclose(aplicar_homogenea(AB, triangulo), aplicar_homogenea(BA, triangulo))

# Dos rotaciones sí conmutan
_, _, distintas = verificar_orden_importa(rotacion_homogenea(30), rotacion_homogenea(60))
assert not distintas

# Validaciones de entrada
for malo in (np.eye(2), np.ones((3, 3)), [[1, 0, 0], [0, 1, 0], [0, 0, np.nan]]):
    try:
        aplicar_homogenea(malo, triangulo)
        raise AssertionError("Debió lanzar ValueError")
    except ValueError:
        pass

try:
    aplicar_homogenea(np.eye(3), [[1, 2], [3, 4]])
    raise AssertionError("Debió lanzar ValueError")
except ValueError:
    pass

# La función de gráfica devuelve el análisis (sin mostrar la ventana)
import matplotlib
matplotlib.use("Agg")
out = graficar_poligono_transformado(matriz_traslacion(2, 1), triangulo, mostrar=False)
assert out["conserva_area"]

print("Todas las pruebas de transformaciones fueron exitosas.")