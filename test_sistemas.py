"""Pruebas del módulo 3 (sistemas de ecuaciones lineales).

Ejecutar desde la raíz del proyecto:  python test_sistemas.py
"""

import numpy as np

from src.sistemas import (
    COMPATIBLE_DETERMINADO,
    COMPATIBLE_INDETERMINADO,
    INCOMPATIBLE,
    clasificar_sistema,
    comparar_con_numpy,
    eliminacion_gauss,
    resolver_sistema,
    validar_sistema,
    verificar_solucion,
)

rng = np.random.default_rng(42)  # semilla fija: pruebas reproducibles


def comprobar_unico(A, b):
    """Verifica un sistema compatible determinado contra numpy.linalg.solve."""
    r = eliminacion_gauss(A, b)
    assert r.clasificacion == COMPATIBLE_DETERMINADO, r.clasificacion
    esperado = np.linalg.solve(np.array(A, float), np.array(b, float))
    assert np.allclose(r.solucion, esperado), (r.solucion, esperado)
    assert verificar_solucion(A, b, r.solucion)
    assert r.rango_A == r.rango_ampliada == len(b)
    return r


def comprobar_indeterminado(A, b, grados):
    """Verifica que la parametrización satisfaga Ax = b para varios t."""
    r = eliminacion_gauss(A, b)
    assert r.clasificacion == COMPATIBLE_INDETERMINADO, r.clasificacion
    assert len(r.variables_libres) == grados
    assert r.direcciones.shape == (grados, np.array(A).shape[1])
    assert verificar_solucion(A, b, r.particular)
    for _ in range(5):
        t = rng.uniform(-10, 10, size=grados)
        assert verificar_solucion(A, b, r.solucion_para(t))
    assert len(r.parametrizacion()) == np.array(A).shape[1]
    return r


# ---------------------------------------------------------------- 2x2 ------
r = comprobar_unico([[2, 1], [1, -1]], [5, 1])
assert np.allclose(r.solucion, [2, 1])

# ---------------------------------------------------------------- 3x3 ------
r = comprobar_unico([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3])
assert np.allclose(r.solucion, [2, 3, -1])

# Pivote cero en la diagonal: requiere intercambio de filas.
r = comprobar_unico([[0, 2, 1], [1, 1, 1], [2, 1, 3]], [5, 6, 13])
assert np.allclose(r.solucion, [4, 1, 1]) or verificar_solucion(
    [[0, 2, 1], [1, 1, 1], [2, 1, 3]], [5, 6, 13], r.solucion
)

# ---------------------------------------------------------------- 4x4 ------
A4 = [[1, 1, 1, 1], [2, -1, 0, 3], [0, 1, -1, 2], [3, 0, 2, -1]]
r = comprobar_unico(A4, [10, 9, 3, 7])

for _ in range(20):  # sistemas aleatorios 3x3 y 4x4 contra NumPy
    for n in (3, 4):
        A = rng.integers(-9, 10, size=(n, n)).astype(float)
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        comprobar_unico(A, rng.integers(-9, 10, size=n))

# ------------------------------------------------ infinitas soluciones -----
r = comprobar_indeterminado([[1, 2], [2, 4]], [3, 6], 1)
assert r.rango_A == 1 and np.allclose(r.particular, [3, 0])
assert np.allclose(r.direcciones[0], [-2, 1])

r = comprobar_indeterminado([[1, 1, 1], [1, 2, 3], [2, 3, 4]], [6, 14, 20], 1)
assert r.rango_A == 2 and r.variables_libres == [2]

comprobar_indeterminado([[1, 2, 3], [2, 4, 6], [3, 6, 9]], [1, 2, 3], 2)
comprobar_indeterminado(
    [[1, 1, 0, 1], [0, 1, 1, 0], [1, 2, 1, 1], [2, 3, 1, 2]], [3, 2, 5, 8], 2
)
comprobar_indeterminado([[1, 2, 3], [4, 5, 6]], [6, 15], 1)  # 2x3

# ---------------------------------------------------- incompatibles --------
assert clasificar_sistema([[1, 2], [2, 4]], [3, 8]) == INCOMPATIBLE
assert clasificar_sistema([[1, 1, 1], [1, 1, 1], [0, 1, 0]], [1, 2, 3]) == INCOMPATIBLE
r = eliminacion_gauss(
    [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 2, 2, 1]], [1, 2, 3, 20]
)
assert r.clasificacion == INCOMPATIBLE
assert r.rango_A == 3 and r.rango_ampliada == 4
assert r.solucion is None and r.particular is None
assert clasificar_sistema([[1, 0], [0, 1], [1, 1]], [1, 1, 5]) == INCOMPATIBLE  # 3x2

# resolver_sistema ya no lanza error: describe cada caso
assert resolver_sistema([[1, 2], [2, 4]], [3, 6]).clasificacion == COMPATIBLE_INDETERMINADO
assert resolver_sistema([[1, 2], [2, 4]], [3, 8]).clasificacion == INCOMPATIBLE

# ------------------------------------------- homogéneos / no homogéneos ----
r = eliminacion_gauss([[1, 2], [3, 4]], [0, 0])
assert r.es_homogeneo and np.allclose(r.solucion, [0, 0])  # solo la trivial

r = comprobar_indeterminado([[1, 2, 3], [2, 4, 6], [1, 0, 1]], [0, 0, 0], 1)
assert r.es_homogeneo and np.allclose(r.particular, 0)

assert not eliminacion_gauss([[1, 0], [0, 1]], [1, 2]).es_homogeneo

# ------------------------------------- verificación contra NumPy -----------
for n in (2, 3, 4):
    A = rng.integers(-9, 10, size=(n, n)).astype(float) + n * np.eye(n) * 5
    cmp = comparar_con_numpy(A, rng.integers(-9, 10, size=n))
    assert cmp["coincide"] and cmp["metodo"] == "numpy.linalg.solve"
    assert cmp["error_maximo"] < 1e-8

for A_s, b_s in [([[1, 2], [2, 4]], [3, 6]), ([[1, 2], [2, 4]], [3, 8]),
                 ([[1, 2, 3], [4, 5, 6]], [6, 15]), ([[1, 0], [0, 1], [1, 1]], [1, 1, 2])]:
    cmp = comparar_con_numpy(A_s, b_s)
    assert cmp["coincide"] and cmp["metodo"] == "numpy.linalg.matrix_rank"

# numpy.linalg.solve falla con matrices singulares; Gauss las clasifica.
try:
    np.linalg.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([3.0, 6.0]))
    raise AssertionError("NumPy debió fallar con una matriz singular.")
except np.linalg.LinAlgError:
    pass

# ---------------------------------------------- validación de entrada ------
entradas_invalidas = [
    ([[1, 2], [3, 4]], [1, 2, 3]),          # b con tamaño incorrecto
    ([1, 2, 3], [1, 2, 3]),                  # A no es matriz
    ([[1, 2], [3, 4]], [[1], [2]]),          # b no es vector
    ([[1, 2], [3]], [1, 2]),                 # filas de distinta longitud
    ([[1, 2], [3, "x"]], [1, 2]),            # valores no numéricos
    ([[1, np.nan], [3, 4]], [1, 2]),         # valores no finitos
    ([[1, np.inf], [3, 4]], [1, 2]),
    (np.zeros((0, 2)), np.zeros(0)),         # sistema vacío
]
for A_mala, b_mala in entradas_invalidas:
    try:
        validar_sistema(A_mala, b_mala)
        raise AssertionError(f"Debió rechazarse: {A_mala}, {b_mala}")
    except ValueError:
        pass

for tol in (0, -1e-3):
    try:
        eliminacion_gauss([[1, 0], [0, 1]], [1, 1], tolerancia=tol)
        raise AssertionError("Debió rechazarse la tolerancia.")
    except ValueError:
        pass

print("Todas las pruebas de sistemas fueron exitosas.")