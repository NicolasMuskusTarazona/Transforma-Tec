
import numpy as np


def validar_matriz(matriz):
    """Valida que el dato recibido sea una matriz numérica."""
    matriz = np.asarray(matriz, dtype=float)

    if matriz.ndim != 2:
        raise ValueError("El dato debe ser una matriz bidimensional.")

    if matriz.size == 0:
        raise ValueError("La matriz no puede estar vacía.")

    if not np.all(np.isfinite(matriz)):
        raise ValueError("La matriz contiene valores no finitos.")

    return matriz


def validar_dimensiones_iguales(a, b):
    """Comprueba que dos matrices tengan las mismas dimensiones."""
    a = validar_matriz(a)
    b = validar_matriz(b)

    if a.shape != b.shape:
        raise ValueError("Las matrices deben tener las mismas dimensiones.")

    return a, b


def sumar_matrices(a, b):
    a, b = validar_dimensiones_iguales(a, b)
    return a + b


def restar_matrices(a, b):
    a, b = validar_dimensiones_iguales(a, b)
    return a - b


def multiplicar_matrices(a, b):
    a = validar_matriz(a)
    b = validar_matriz(b)

    if a.shape[1] != b.shape[0]:
        raise ValueError("Las dimensiones no permiten multiplicar las matrices.")

    return np.matmul(a, b)


def multiplicar_matriz_escalar(matriz, escalar):
    matriz = validar_matriz(matriz)
    return matriz * escalar


def determinante_matriz(matriz):
    matriz = validar_matriz(matriz)

    if matriz.shape[0] != matriz.shape[1]:
        raise ValueError("El determinante requiere una matriz cuadrada.")

    return np.linalg.det(matriz)


def inversa_matriz(matriz):
    matriz = validar_matriz(matriz)

    if matriz.shape[0] != matriz.shape[1]:
        raise ValueError("La inversa requiere una matriz cuadrada.")

    if np.isclose(np.linalg.det(matriz), 0, atol=1e-10):
        raise ValueError("La matriz es singular y no tiene inversa.")

    return np.linalg.inv(matriz)


def rango_matriz(matriz):
    matriz = validar_matriz(matriz)
    return np.linalg.matrix_rank(matriz)


def matriz_identidad(dimension):
    if not isinstance(dimension, (int, np.integer)) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("La dimensión debe ser un entero positivo.")

    return np.eye(dimension)


def componer_matrices(a, b):
    """Compone dos transformaciones matriciales mediante su producto."""
    return multiplicar_matrices(a, b)


def graficar_matriz(matriz, titulo="Visualización de matriz"):
    import matplotlib.pyplot as plt

    matriz = validar_matriz(matriz)

    plt.figure(figsize=(7, 5))
    plt.imshow(matriz, cmap="viridis", aspect="auto")
    plt.colorbar(label="Valor")
    plt.title(titulo)
    plt.xlabel("Columnas")
    plt.ylabel("Filas")

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            plt.text(j, i, f"{matriz[i, j]:.1f}",
                     ha="center", va="center", color="white")

    plt.show()
