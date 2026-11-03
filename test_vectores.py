
import numpy as np

from src.vectores import (
    sumar_vectores,
    restar_vectores,
    multiplicar_escalar,
    producto_punto,
    producto_cruz,
    norma_vector,
    angulo_entre_vectores
)

a = np.array([3, 4])
b = np.array([-2, 3])

assert np.allclose(sumar_vectores(a, b), [1, 7])
assert np.allclose(restar_vectores(a, b), [5, 1])
assert np.allclose(multiplicar_escalar(a, 2), [6, 8])
assert producto_punto(a, b) == 6
assert norma_vector(a) == 5
assert np.isclose(angulo_entre_vectores(a, b), 70.56, atol=0.01)
cruz = producto_cruz([1, 0, 0], [0, 1, 0])
assert np.allclose(cruz, [0, 0, 1])

print("Todas las pruebas de vectores fueron exitosas.")
