
import numpy as np

from src.transformaciones import (
    aplicar_transformacion,
    matriz_rotacion,
    matriz_escalamiento,
    matriz_reflexion,
    matriz_cizallamiento,
    componer_transformaciones,
    es_transformacion_lineal
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

print("Todas las pruebas de transformaciones fueron exitosas.")
