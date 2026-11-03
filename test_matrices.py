
import numpy as np

from src.matrices import (
    sumar_matrices,
    restar_matrices,
    multiplicar_matrices,
    multiplicar_matriz_escalar,
    determinante_matriz,
    inversa_matriz,
    rango_matriz,
    matriz_identidad,
    componer_matrices
)

A = np.array([
    [2, 1],
    [3, 4]
])

B = np.array([
    [1, 2],
    [0, 3]
])

assert np.allclose(sumar_matrices(A, B), [[3, 3], [3, 7]])

assert np.allclose(restar_matrices(A, B), [[1, -1], [3, 1]])

assert np.allclose(multiplicar_matrices(A, B), [[2, 7], [3, 18]])

assert np.allclose(multiplicar_matriz_escalar(A, 2), [[4, 2], [6, 8]])

assert np.isclose(determinante_matriz(A), 5)

assert np.allclose(
    inversa_matriz(A),
    [[0.8, -0.2], [-0.6, 0.4]]
)

assert rango_matriz(A) == 2

assert np.allclose(matriz_identidad(2), np.eye(2))

assert np.allclose(componer_matrices(A, B), A @ B)

# Prueba de matriz singular
try:
    inversa_matriz([[1, 2], [2, 4]])
    raise AssertionError("Debió rechazarse la matriz singular.")
except ValueError:
    pass

print("Todas las pruebas de matrices fueron exitosas.")
