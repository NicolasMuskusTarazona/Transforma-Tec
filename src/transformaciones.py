
import numpy as np


def validar_transformacion(matriz, vector):
    """Valida la matriz de transformación y el vector."""
    matriz = np.asarray(matriz, dtype=float)
    vector = np.asarray(vector, dtype=float)

    if matriz.ndim != 2 or vector.ndim != 1:
        raise ValueError("Se requiere una matriz y un vector.")

    if matriz.shape[1] != vector.size:
        raise ValueError("Las dimensiones de la matriz y el vector no coinciden.")

    if matriz.size == 0 or vector.size == 0:
        raise ValueError("Los datos no pueden estar vacíos.")

    if not np.all(np.isfinite(matriz)) or not np.all(np.isfinite(vector)):
        raise ValueError("Los datos deben ser finitos.")

    return matriz, vector


def aplicar_transformacion(matriz, vector):
    matriz, vector = validar_transformacion(matriz, vector)
    return matriz @ vector


def matriz_rotacion(angulo_grados):
    """Crea una matriz de rotación en 2D."""
    angulo = np.radians(angulo_grados)

    return np.array([
        [np.cos(angulo), -np.sin(angulo)],
        [np.sin(angulo), np.cos(angulo)]
    ])


def matriz_escalamiento(sx, sy):
    """Crea una matriz de escalamiento en 2D."""
    return np.array([
        [sx, 0],
        [0, sy]
    ], dtype=float)


def matriz_reflexion(eje="x"):
    """Crea una matriz de reflexión respecto al eje X o Y."""
    if eje == "x":
        return np.array([[1, 0], [0, -1]])

    if eje == "y":
        return np.array([[-1, 0], [0, 1]])

    raise ValueError("El eje debe ser 'x' o 'y'.")


def matriz_cizallamiento(kx=0, ky=0):
    """Crea una matriz de cizallamiento en 2D."""
    return np.array([
        [1, kx],
        [ky, 1]
    ], dtype=float)


def componer_transformaciones(*matrices):
    """Compone transformaciones en el orden en que se aplican."""
    if not matrices:
        raise ValueError("Debe proporcionar al menos una matriz.")

    resultado = np.asarray(matrices[0], dtype=float)

    if resultado.ndim != 2 or resultado.shape[0] != resultado.shape[1]:
        raise ValueError("Las transformaciones deben ser matrices cuadradas.")

    for matriz in matrices[1:]:
        matriz = np.asarray(matriz, dtype=float)

        if matriz.shape != resultado.shape:
            raise ValueError("Las matrices deben tener las mismas dimensiones.")

        resultado = matriz @ resultado

    return resultado


def es_transformacion_lineal(matriz):
    """Comprueba si una transformación matricial es válida."""
    matriz = np.asarray(matriz, dtype=float)

    return (
        matriz.ndim == 2
        and matriz.size > 0
        and np.all(np.isfinite(matriz))
    )


def graficar_transformacion(matriz, puntos, titulo="Transformación lineal"):
    import matplotlib.pyplot as plt

    matriz = np.asarray(matriz, dtype=float)
    puntos = np.asarray(puntos, dtype=float)

    if matriz.shape != (2, 2):
        raise ValueError("La gráfica requiere una matriz 2x2.")

    if puntos.ndim != 2 or puntos.shape[1] != 2 or puntos.shape[0] == 0:
        raise ValueError("Los puntos deben tener dos coordenadas.")

    if not np.all(np.isfinite(matriz)) or not np.all(np.isfinite(puntos)):
        raise ValueError("Los datos deben ser finitos.")

    transformados = (matriz @ puntos.T).T

    plt.figure(figsize=(8, 7))

    plt.scatter(puntos[:, 0], puntos[:, 1], label="Originales")
    plt.scatter(transformados[:, 0], transformados[:, 1],
                label="Transformados")

    for original, nuevo in zip(puntos, transformados):
        plt.plot(
            [original[0], nuevo[0]],
            [original[1], nuevo[1]],
            linestyle="--"
        )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(True)
    plt.axis("equal")
    plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

    return transformados
