
import numpy as np


def validar_sistema(A, b):
    """Valida la matriz de coeficientes y el vector independiente."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or b.ndim != 1:
        raise ValueError("A debe ser una matriz y b un vector.")

    if A.shape[0] != b.size:
        raise ValueError("El número de ecuaciones debe coincidir con b.")

    if A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError("El sistema no puede estar vacío.")

    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("El sistema contiene valores no finitos.")

    return A, b


def eliminacion_gauss(A, b, tolerancia=1e-10):
    """Resuelve sistemas cuadrados mediante eliminación de Gauss."""
    A, b = validar_sistema(A, b)

    if A.shape[0] != A.shape[1]:
        raise ValueError("La eliminación de Gauss requiere un sistema cuadrado.")

    n = len(b)
    matriz = np.column_stack((A, b)).astype(float)

    for i in range(n):
        pivote = i + np.argmax(np.abs(matriz[i:, i]))

        if abs(matriz[pivote, i]) < tolerancia:
            raise ValueError("El sistema no tiene solución única.")

        matriz[[i, pivote]] = matriz[[pivote, i]]

        for j in range(i + 1, n):
            factor = matriz[j, i] / matriz[i, i]
            matriz[j] -= factor * matriz[i]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            matriz[i, -1] -
            np.dot(matriz[i, i + 1:n], x[i + 1:n])
        ) / matriz[i, i]

    return x


def clasificar_sistema(A, b, tolerancia=1e-10):
    """Clasifica un sistema según sus rangos."""
    A, b = validar_sistema(A, b)

    rango_A = np.linalg.matrix_rank(A, tol=tolerancia)
    ampliada = np.column_stack((A, b))
    rango_ampliada = np.linalg.matrix_rank(ampliada, tol=tolerancia)

    if rango_A != rango_ampliada:
        return "Incompatible (sin solución)"

    if rango_A == A.shape[1]:
        return "Compatible determinado (solución única)"

    return "Compatible indeterminado (infinitas soluciones)"


def resolver_sistema(A, b):
    """Resuelve sistemas compatibles."""
    A, b = validar_sistema(A, b)

    clasificacion = clasificar_sistema(A, b)

    if clasificacion.startswith("Incompatible"):
        raise ValueError("El sistema no tiene solución.")

    if clasificacion.startswith("Compatible indeterminado"):
        raise ValueError("El sistema tiene infinitas soluciones.")

    return np.linalg.solve(A, b)


def verificar_solucion(A, b, x, tolerancia=1e-8):
    """Comprueba si Ax = b."""
    A, b = validar_sistema(A, b)
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or x.size != A.shape[1]:
        raise ValueError("La solución tiene dimensiones incorrectas.")

    return np.allclose(A @ x, b, atol=tolerancia)


def graficar_sistema(A, b):
    """Representa gráficamente un sistema de dos variables."""
    import matplotlib.pyplot as plt

    A, b = validar_sistema(A, b)

    if A.shape != (2, 2):
        raise ValueError("La gráfica requiere dos ecuaciones y dos variables.")

    x = np.linspace(-10, 10, 400)

    plt.figure(figsize=(8, 6))

    for i in range(2):
        if np.isclose(A[i, 1], 0):
            if np.isclose(A[i, 0], 0):
                continue

            x_vertical = b[i] / A[i, 0]
            plt.axvline(x_vertical, label=f"Ecuación {i + 1}")
        else:
            y = (b[i] - A[i, 0] * x) / A[i, 1]
            plt.plot(x, y, label=f"Ecuación {i + 1}")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Representación de un sistema lineal")
    plt.grid(True)
    plt.legend()
    plt.show()
