import numpy as np

from src.espacios_vectoriales import (
    verificar_independencia,
    calcular_dimension,
    obtener_base,
    pertenece_al_espacio,
    calcular_coordenadas
)

# Vectores linealmente independientes
vectores = np.array([
    [1, 0],
    [0, 1]
])

assert verificar_independencia(vectores)
assert calcular_dimension(vectores) == 2

# Vectores linealmente dependientes
dependientes = np.array([
    [1, 2],
    [2, 4],
    [3, 6]
])

assert not verificar_independencia(dependientes)
assert calcular_dimension(dependientes) == 1

# Obtener una base
generadores = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
])

base = obtener_base(generadores)

assert base.shape[0] == 2
assert verificar_independencia(base)

# Pertenencia al espacio generado
assert pertenece_al_espacio([3, 5], generadores)

# Coordenadas respecto a una base
base_2 = np.array([
    [1, 0],
    [0, 1]
])

coordenadas = calcular_coordenadas([3, 5], base_2)

assert np.allclose(coordenadas, [3, 5])

print("Todas las pruebas de espacios vectoriales fueron exitosas.")

# ---------------------------------------------------------------------------
# Espacio nulo y espacio columna
# ---------------------------------------------------------------------------
from src.espacios_vectoriales import (
    espacio_nulo,
    espacio_columna,
    verificar_rango_nulidad,
)

# Caso 3x4 de rango 2 (la fila 3 es la suma de las filas 1 y 2): nulidad 2
A = np.array([
    [1, 2, 1, 3],
    [2, 4, 0, 2],
    [3, 6, 1, 5]
], dtype=float)

nulo = espacio_nulo(A)
columna = espacio_columna(A)

assert nulo.shape == (2, 4)
assert np.allclose(A @ nulo.T, 0)                # cada vector base cumple Ax = 0
assert verificar_independencia(nulo)             # los vectores base son independientes
assert columna.shape == (2, 3)
assert verificar_independencia(columna)
assert calcular_dimension(columna) == np.linalg.matrix_rank(A)
assert pertenece_al_espacio(A[:, 1], columna)    # toda columna de A está en C(A)
assert pertenece_al_espacio(A @ [1, -2, 3, 0.5], columna)
assert not pertenece_al_espacio([1, 0, 0], columna)

# Teorema rango-nulidad
r = verificar_rango_nulidad(A)
assert r["rango"] == 2 and r["nulidad"] == 2 and r["columnas"] == 4 and r["cumple"]

# Matriz invertible: N(A) = {0} y C(A) = R^n
I3 = np.array([[2, 1, 0], [0, 1, 3], [1, 0, 1]], dtype=float)
assert espacio_nulo(I3).shape == (0, 3)
assert espacio_columna(I3).shape == (3, 3)
assert verificar_rango_nulidad(I3)["cumple"]

# Pivote cero en la diagonal: las columnas base deben ser columnas ORIGINALES de A
P = np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]], dtype=float)
assert verificar_rango_nulidad(P)["rango"] == 1
assert np.allclose(espacio_columna(P), [[1, 2, 3]])
assert np.allclose(P @ espacio_nulo(P).T, 0)
assert espacio_nulo(P).shape == (2, 3)

# Matriz nula: N(A) = R^n y C(A) = {0}
Z = np.zeros((2, 3))
assert espacio_nulo(Z).shape == (3, 3)
assert espacio_columna(Z).shape == (0, 2)
assert verificar_rango_nulidad(Z)["cumple"]

# Matrices aleatorias (semilla fija): el teorema rango-nulidad siempre se cumple
rng = np.random.default_rng(42)
for _ in range(50):
    M = rng.integers(-3, 4, size=(rng.integers(1, 6), rng.integers(1, 6))).astype(float)
    assert verificar_rango_nulidad(M)["cumple"]
    assert np.allclose(M @ espacio_nulo(M).T, 0)
    assert verificar_rango_nulidad(M)["rango"] == np.linalg.matrix_rank(M)

# Validación de entradas
for invalida in ([1, 2, 3], [[np.nan, 1], [1, 1]], np.zeros((0, 2))):
    try:
        espacio_nulo(invalida)
        assert False, "debió lanzar ValueError"
    except ValueError:
        pass

print("Todas las pruebas de espacio nulo y espacio columna fueron exitosas.")