
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
