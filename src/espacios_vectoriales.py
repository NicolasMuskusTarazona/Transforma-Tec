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

# ---------------------------------------------------------------------------
# Espacio nulo y espacio columna
# ---------------------------------------------------------------------------
def espacio_nulo(A, tolerancia=1e-10):
    """Calcula una base del espacio nulo N(A) = {x : Ax = 0}.

    Se resuelve el sistema homogéneo Ax = 0 con la eliminación de Gauss del
    módulo 3: cada variable libre produce un vector director, y esos vectores
    forman una base de N(A).

    Parámetros
    ----------
    A : array_like de forma (m, n)
    tolerancia : float
        Umbral para considerar un valor igual a cero.

    Devuelve
    --------
    ndarray de forma (nulidad, n); cada fila es un vector de la base.
    Si N(A) = {0} devuelve un arreglo de forma (0, n).
    """
    from src.sistemas import eliminacion_gauss

    A = validar_vectores(A)
    resultado = eliminacion_gauss(A, np.zeros(A.shape[0]), tolerancia)

    if not resultado.variables_libres:
        return np.zeros((0, A.shape[1]))

    return resultado.direcciones


def espacio_columna(A, tolerancia=1e-10):
    """Calcula una base del espacio columna C(A) = {Ax : x en R^n}.

    Una base de C(A) está formada por las columnas de A que corresponden a las
    columnas pivote de la eliminación de Gauss (las columnas originales, no las
    de la matriz escalonada).

    Devuelve
    --------
    ndarray de forma (rango, m); cada fila es una columna de A.
    Si A = 0 devuelve un arreglo de forma (0, m).
    """
    from src.sistemas import eliminacion_gauss

    A = validar_vectores(A)
    resultado = eliminacion_gauss(A, np.zeros(A.shape[0]), tolerancia)

    return A[:, resultado.pivotes].T


def verificar_rango_nulidad(A, tolerancia=1e-10):
    """Comprueba el teorema rango-nulidad: rango(A) + nulidad(A) = n.

    Devuelve un diccionario con las claves ``rango``, ``nulidad``,
    ``columnas`` (n) y ``cumple`` (bool).
    """
    A = validar_vectores(A)

    rango = espacio_columna(A, tolerancia).shape[0]
    nulidad = espacio_nulo(A, tolerancia).shape[0]
    n = A.shape[1]

    return {
        "rango": rango,
        "nulidad": nulidad,
        "columnas": n,
        "cumple": rango + nulidad == n,
    }


def _dibujar_subespacio(ax, base, color, etiqueta, limite):
    """Dibuja en 2D el subespacio generado por ``base`` (punto, recta o plano)."""
    if base.shape[0] == 0:
        ax.scatter(0, 0, color=color, s=80, zorder=5,
                   label=f"{etiqueta} = {{0}} (solo el origen)")
    elif base.shape[0] == 1:
        v = base[0] / np.linalg.norm(base[0])
        ax.plot([-limite * v[0], limite * v[0]],
                [-limite * v[1], limite * v[1]],
                color=color, linewidth=2, label=etiqueta)
    else:
        ax.axvspan(-limite, limite, color=color, alpha=0.15,
                   label=f"{etiqueta} = todo R²")


def graficar_espacio_nulo_y_columna(A, guardar_en=None):
    """Grafica N(A) en el dominio y C(A) en el codominio de una matriz 2x2.

    Panel izquierdo: espacio nulo en R^2 (entradas x que A envía a 0), con sus
    vectores base. Panel derecho: espacio columna en R^2 (salidas Ax), con las
    columnas de A como flechas.
    """
    import matplotlib.pyplot as plt

    A = validar_vectores(A)

    if A.shape != (2, 2):
        raise ValueError("La gráfica requiere una matriz de 2x2.")

    nulo = espacio_nulo(A)
    columna = espacio_columna(A)
    limite = max(3.0, 1.3 * float(np.max(np.abs(A))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Espacio nulo (dominio) ---
    _dibujar_subespacio(ax1, nulo, "tab:green", "N(A)", limite)
    for k, v in enumerate(nulo):
        ax1.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
                   color="darkgreen", zorder=4)
        ax1.annotate(f"n{k + 1} = ({v[0]:.3g}, {v[1]:.3g})\nA·n{k + 1} = 0",
                     (v[0], v[1]), xytext=(-25, -45), textcoords="offset points",
                     fontsize=9, color="darkgreen",
                     bbox=dict(boxstyle="round", facecolor="white",
                               edgecolor="darkgreen"),
                     arrowprops=dict(arrowstyle="-", color="darkgreen"))
    ax1.set_title("Espacio nulo N(A) en el dominio (x₁, x₂)")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")

    # --- Espacio columna (codominio) ---
    _dibujar_subespacio(ax2, columna, "tab:blue", "C(A)", limite)
    colores = ["tab:red", "tab:orange"]
    for j in range(2):
        a = A[:, j]
        # Anchos distintos: si las columnas son colineales, ninguna tapa a la otra.
        ax2.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1,
                   color=colores[j], zorder=4 - j, width=0.006 + 0.008 * j,
                   label=f"a{j + 1} = ({a[0]:.3g}, {a[1]:.3g})")
    ax2.set_title("Espacio columna C(A) en el codominio (y₁, y₂)")
    ax2.set_xlabel("y₁")
    ax2.set_ylabel("y₂")

    for ax in (ax1, ax2):
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-limite, limite)
        ax.set_ylim(-limite, limite)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    if guardar_en:
        fig.savefig(guardar_en, dpi=150, bbox_inches="tight")
    plt.show()