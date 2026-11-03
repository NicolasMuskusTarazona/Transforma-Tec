
import numpy as np


def validar_vectores(vectores):
    """Valida una colección de vectores de igual dimensión."""
    vectores = np.asarray(vectores, dtype=float)

    if vectores.ndim != 2:
        raise ValueError("Se requiere una colección bidimensional de vectores.")

    if vectores.shape[0] == 0 or vectores.shape[1] == 0:
        raise ValueError("La colección no puede estar vacía.")

    if not np.all(np.isfinite(vectores)):
        raise ValueError("Los vectores deben contener valores finitos.")

    return vectores


def verificar_independencia(vectores, tolerancia=1e-10):
    """Determina si los vectores son linealmente independientes."""
    vectores = validar_vectores(vectores)

    matriz = vectores.T
    rango = np.linalg.matrix_rank(matriz, tol=tolerancia)

    return rango == vectores.shape[0]


def calcular_dimension(vectores, tolerancia=1e-10):
    """Calcula la dimensión del espacio generado por los vectores."""
    vectores = validar_vectores(vectores)

    return np.linalg.matrix_rank(vectores.T, tol=tolerancia)


def obtener_base(vectores, tolerancia=1e-10):
    """Obtiene una base a partir de un conjunto generador."""
    vectores = validar_vectores(vectores)

    _, indices = np.linalg.qr(vectores.T, mode="reduced")

    # Se seleccionan columnas independientes mediante rango incremental.
    seleccionados = []
    rango_actual = 0

    for i in range(vectores.shape[0]):
        candidato = seleccionados + [i]
        rango_nuevo = np.linalg.matrix_rank(
            vectores[candidato].T,
            tol=tolerancia
        )

        if rango_nuevo > rango_actual:
            seleccionados.append(i)
            rango_actual = rango_nuevo

    return vectores[seleccionados]


def pertenece_al_espacio(vector, generadores, tolerancia=1e-8):
    """Comprueba si un vector pertenece al espacio generado."""
    generadores = validar_vectores(generadores)
    vector = np.asarray(vector, dtype=float)

    if vector.ndim != 1 or vector.size != generadores.shape[1]:
        raise ValueError("El vector tiene dimensiones incorrectas.")

    if not np.all(np.isfinite(vector)):
        raise ValueError("El vector debe contener valores finitos.")

    coeficientes, _, _, _ = np.linalg.lstsq(
        generadores.T,
        vector,
        rcond=None
    )

    reconstruido = generadores.T @ coeficientes

    return np.allclose(reconstruido, vector, atol=tolerancia)


def calcular_coordenadas(vector, base, tolerancia=1e-8):
    """Calcula las coordenadas de un vector respecto a una base."""
    base = validar_vectores(base)
    vector = np.asarray(vector, dtype=float)

    if vector.ndim != 1 or vector.size != base.shape[1]:
        raise ValueError("El vector tiene dimensiones incorrectas.")

    if not verificar_independencia(base):
        raise ValueError("Los vectores proporcionados no forman una base.")

    coeficientes, _, _, _ = np.linalg.lstsq(
        base.T,
        vector,
        rcond=None
    )

    if not np.allclose(base.T @ coeficientes, vector, atol=tolerancia):
        raise ValueError("El vector no pertenece al espacio generado por la base.")

    return coeficientes


def graficar_espacio(generadores, titulo="Espacio generado"):
    """Visualiza vectores generadores en dos dimensiones."""
    import matplotlib.pyplot as plt

    generadores = validar_vectores(generadores)

    if generadores.shape[1] != 2:
        raise ValueError("La visualización requiere vectores 2D.")

    plt.figure(figsize=(7, 7))

    for i, vector in enumerate(generadores):
        plt.quiver(
            0, 0,
            vector[0], vector[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            label=f"Vector {i + 1}"
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
