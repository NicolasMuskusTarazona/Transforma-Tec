
import numpy as np

from src.sistemas import (
    eliminacion_gauss,
    clasificar_sistema,
    resolver_sistema,
    verificar_solucion
)

# Sistema con solución única
A = np.array([
    [2, 1],
    [1, -1]
])

b = np.array([5, 1])

solucion = eliminacion_gauss(A, b)

assert np.allclose(solucion, [2, 1])
assert verificar_solucion(A, b, solucion)

assert clasificar_sistema(A, b).startswith(
    "Compatible determinado"
)

# Sistema con infinitas soluciones
A2 = [[1, 2], [2, 4]]
b2 = [3, 6]

assert clasificar_sistema(A2, b2).startswith(
    "Compatible indeterminado"
)

# Sistema incompatible
A3 = [[1, 2], [2, 4]]
b3 = [3, 8]

assert clasificar_sistema(A3, b3).startswith(
    "Incompatible"
)

# Comprobar rechazo de sistemas sin solución única
try:
    resolver_sistema(A3, b3)
    raise AssertionError("Debió rechazarse el sistema incompatible.")
except ValueError:
    pass

print("Todas las pruebas de sistemas fueron exitosas.")
