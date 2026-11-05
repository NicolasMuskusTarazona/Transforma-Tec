"""Pruebas del módulo de aplicaciones (Ingeniería de Software).

Ejecutar desde la raíz del proyecto:  python test_aplicaciones.py
"""

import numpy as np

from src.aplicaciones import (
    analizar_flujo_red,
    balancear_carga,
    sistema_balance_carga,
    sistema_flujo_red,
)
from src.sistemas import (
    COMPATIBLE_DETERMINADO,
    COMPATIBLE_INDETERMINADO,
    INCOMPATIBLE,
    comparar_con_numpy,
    eliminacion_gauss,
    verificar_solucion,
)

# --- Balance de carga con 3 servidores (capacidades 4, 2 y 5) ---------------
r = balancear_carga(1100, [4, 2, 5])
assert r.clasificacion == COMPATIBLE_DETERMINADO and not r.es_homogeneo
assert np.allclose(r.solucion, [400, 200, 500])
assert np.isclose(r.solucion.sum(), 1100)

# --- 4 servidores (sistema 4x4) y comparación con NumPy ---------------------
pesos = [3, 2, 1, 4]
A, b = sistema_balance_carga(1000, pesos)
assert A.shape == (4, 4)
r = balancear_carga(1000, pesos)
assert np.allclose(r.solucion, 1000 * np.array(pesos) / sum(pesos))
assert np.allclose(r.solucion, np.linalg.solve(A, b))
assert comparar_con_numpy(A, b)["coincide"]

# --- Caso homogéneo (carga total 0): solo la solución trivial ---------------
r = balancear_carga(0, [4, 2, 5])
assert r.es_homogeneo and np.allclose(r.solucion, 0)

# --- Política incompleta: sin ecuación de un servidor -> infinitas soluciones
A, b = sistema_balance_carga(1100, [4, 2, 5])
r = eliminacion_gauss(A[:-1], b[:-1])
assert r.clasificacion == COMPATIBLE_INDETERMINADO and len(r.variables_libres) == 1
assert verificar_solucion(A[:-1], b[:-1], r.solucion_para([300.0]))

# --- Requisitos contradictorios: dos totales distintos -> incompatible ------
A, b = sistema_balance_carga(1100, [4, 2, 5])
A_c = np.vstack([A, np.ones(3)])
b_c = np.append(b, 900.0)
assert eliminacion_gauss(A_c, b_c).clasificacion == INCOMPATIBLE

# --- Red de enrutadores: sistema homogéneo con circulación de flujo ---------
A, b = sistema_flujo_red()
r = analizar_flujo_red()
assert r.es_homogeneo and r.clasificacion == COMPATIBLE_INDETERMINADO
assert r.rango_A == 2 and len(r.variables_libres) == 2
assert np.allclose(r.particular, 0)
for t in ([5, 2], [10, 0], [3, 3]):
    f = r.solucion_para(t)
    assert np.allclose(A @ f, 0)
assert comparar_con_numpy(A, b)["coincide"]

# --- Validación de entradas --------------------------------------------------
for args in [(100, [5]), (100, [1, -2]), (-5, [1, 2]), (100, [[1, 2]]), (100, [1, np.nan])]:
    try:
        sistema_balance_carga(*args)
        raise AssertionError(f"Debió rechazarse: {args}")
    except ValueError:
        pass

print("Todas las pruebas de aplicaciones fueron exitosas.")