
import numpy as np


def validar_vector(vector):
    """Valida que el dato recibido sea un vector numérico."""
    vector = np.asarray(vector, dtype=float)

    if vector.ndim != 1:
        raise ValueError("El dato debe ser un vector unidimensional.")

    if vector.size == 0:
        raise ValueError("El vector no puede estar vacío.")

    return vector


def validar_dimensiones_iguales(a, b):
    """Comprueba que dos vectores tengan la misma dimensión."""
    a = validar_vector(a)
    b = validar_vector(b)

    if a.shape != b.shape:
        raise ValueError("Los vectores deben tener la misma dimensión.")

    return a, b


def sumar_vectores(a, b):
    a, b = validar_dimensiones_iguales(a, b)
    return a + b


def restar_vectores(a, b):
    a, b = validar_dimensiones_iguales(a, b)
    return a - b


def multiplicar_escalar(vector, escalar):
    vector = validar_vector(vector)
    return vector * escalar


def producto_punto(a, b):
    a, b = validar_dimensiones_iguales(a, b)
    return np.dot(a, b)


def producto_cruz(a, b):
    a, b = validar_dimensiones_iguales(a, b)

    if a.size != 3:
        raise ValueError("El producto cruz requiere vectores de 3 dimensiones.")

    return np.cross(a, b)


def norma_vector(vector):
    vector = validar_vector(vector)
    return np.linalg.norm(vector)


def angulo_entre_vectores(a, b):
    a, b = validar_dimensiones_iguales(a, b)

    norma_a = norma_vector(a)
    norma_b = norma_vector(b)

    if norma_a == 0 or norma_b == 0:
        raise ValueError("No se puede calcular el ángulo con un vector nulo.")

    coseno = np.dot(a, b) / (norma_a * norma_b)
    coseno = np.clip(coseno, -1.0, 1.0)

    return np.degrees(np.arccos(coseno))


def graficar_vectores(a, b, titulo="Representación de vectores"):
    import matplotlib.pyplot as plt

    a, b = validar_dimensiones_iguales(a, b)

    if a.size != 2:
        raise ValueError("La gráfica requiere vectores de 2 dimensiones.")

    plt.figure(figsize=(7, 7))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)

    plt.quiver(0, 0, a[0], a[1], angles="xy",
               scale_units="xy", scale=1, label="Vector A")

    plt.quiver(0, 0, b[0], b[1], angles="xy",
               scale_units="xy", scale=1, label="Vector B")

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid(True)
    plt.legend()
    plt.title(titulo)
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.gca().set_aspect("equal")
    plt.show()
