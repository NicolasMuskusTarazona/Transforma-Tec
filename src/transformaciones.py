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


# ======================================================================
# Coordenadas homogéneas (matrices 3x3) y transformación de polígonos
# ======================================================================

def _validar_poligono(poligono):
    """Valida un polígono dado como arreglo (n, 2) con n >= 3 vértices."""
    poligono = np.asarray(poligono, dtype=float)

    if poligono.ndim != 2 or poligono.shape[1] != 2:
        raise ValueError("El polígono debe tener forma (n, 2).")

    if poligono.shape[0] < 3:
        raise ValueError("El polígono requiere al menos 3 vértices.")

    if not np.all(np.isfinite(poligono)):
        raise ValueError("Los vértices deben ser finitos.")

    return poligono


def _validar_homogenea(matriz):
    """Valida que la matriz sea 3x3 y tenga la forma afín [[A, t], [0, 1]]."""
    matriz = np.asarray(matriz, dtype=float)

    if matriz.shape != (3, 3):
        raise ValueError("La matriz homogénea debe ser de 3x3.")

    if not np.all(np.isfinite(matriz)):
        raise ValueError("La matriz debe contener valores finitos.")

    if not np.allclose(matriz[2], [0, 0, 1]):
        raise ValueError("La última fila de la matriz homogénea debe ser [0, 0, 1].")

    return matriz


def matriz_homogenea(matriz_2x2=None, traslacion=(0.0, 0.0)):
    """Construye la matriz afín 3x3 [[A, t], [0, 1]] a partir de A (2x2) y t."""
    if matriz_2x2 is None:
        matriz_2x2 = np.eye(2)

    matriz_2x2 = np.asarray(matriz_2x2, dtype=float)
    traslacion = np.asarray(traslacion, dtype=float)

    if matriz_2x2.shape != (2, 2):
        raise ValueError("La parte lineal debe ser una matriz 2x2.")

    if traslacion.shape != (2,):
        raise ValueError("La traslación debe tener dos componentes.")

    resultado = np.eye(3)
    resultado[:2, :2] = matriz_2x2
    resultado[:2, 2] = traslacion
    return resultado


def matriz_traslacion(tx, ty):
    """Matriz homogénea 3x3 de traslación por el vector (tx, ty)."""
    return matriz_homogenea(np.eye(2), (tx, ty))


def rotacion_homogenea(angulo_grados):
    """Rotación alrededor del origen como matriz homogénea 3x3."""
    return matriz_homogenea(matriz_rotacion(angulo_grados))


def escalamiento_homogeneo(sx, sy):
    """Escalamiento respecto al origen como matriz homogénea 3x3."""
    return matriz_homogenea(matriz_escalamiento(sx, sy))


def reflexion_homogenea(eje="x"):
    """Reflexión respecto al eje X o Y como matriz homogénea 3x3."""
    return matriz_homogenea(matriz_reflexion(eje))


def cizallamiento_homogeneo(kx=0, ky=0):
    """Cizallamiento como matriz homogénea 3x3."""
    return matriz_homogenea(matriz_cizallamiento(kx, ky))


def rotacion_alrededor_de_punto(angulo_grados, centro):
    """Rotación alrededor de un punto C: T(C) · R · T(-C)."""
    cx, cy = centro
    return componer_transformaciones(
        matriz_traslacion(-cx, -cy),
        rotacion_homogenea(angulo_grados),
        matriz_traslacion(cx, cy),
    )


def puntos_a_homogeneos(puntos):
    """Convierte puntos (n, 2) a coordenadas homogéneas (n, 3) con w = 1."""
    puntos = np.asarray(puntos, dtype=float)

    if puntos.ndim != 2 or puntos.shape[1] != 2:
        raise ValueError("Los puntos deben tener forma (n, 2).")

    return np.column_stack((puntos, np.ones(puntos.shape[0])))


def aplicar_homogenea(matriz, poligono):
    """Aplica una matriz 3x3 a los vértices de un polígono (n, 2).

    Cada vértice (x, y) se escribe como [x, y, 1]^T, se multiplica por la
    matriz y se descarta la tercera coordenada.
    """
    matriz = _validar_homogenea(matriz)
    poligono = _validar_poligono(poligono)

    homogeneos = puntos_a_homogeneos(poligono)
    resultado = (matriz @ homogeneos.T).T
    return resultado[:, :2]


# ----------------------------------------------------------------------
# Polígonos de prueba
# ----------------------------------------------------------------------

def crear_triangulo():
    """Triángulo del proyecto: A(1, 1), B(5, 1), C(3, 4). Área = 6."""
    return np.array([[1, 1], [5, 1], [3, 4]], dtype=float)


def crear_cuadrado(lado=2.0, origen=(1.0, 1.0)):
    """Cuadrado con esquina inferior izquierda en `origen`."""
    if lado <= 0:
        raise ValueError("El lado debe ser positivo.")

    x0, y0 = origen
    return np.array([
        [x0, y0],
        [x0 + lado, y0],
        [x0 + lado, y0 + lado],
        [x0, y0 + lado],
    ], dtype=float)


def crear_sprite():
    """Sprite sencillo en forma de flecha (7 vértices), útil para videojuegos."""
    return np.array([
        [0, 0],
        [3, 0],
        [3, -1],
        [5, 1],
        [3, 3],
        [3, 2],
        [0, 2],
    ], dtype=float)


# ----------------------------------------------------------------------
# Propiedades geométricas
# ----------------------------------------------------------------------

def area_poligono(poligono):
    """Área de un polígono simple mediante la fórmula del cordón (shoelace)."""
    p = _validar_poligono(poligono)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def longitudes_lados(poligono):
    """Longitud de cada lado del polígono, en el orden de los vértices."""
    p = _validar_poligono(poligono)
    return np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)


def analizar_transformacion(matriz, poligono, tolerancia=1e-9):
    """Aplica la matriz 3x3 al polígono y verifica qué propiedades se conservan.

    Devuelve un diccionario con el polígono transformado, el determinante de
    la parte lineal, las áreas, la razón de áreas y si se conservan el área y
    las longitudes de los lados.
    """
    matriz = _validar_homogenea(matriz)
    poligono = _validar_poligono(poligono)
    transformado = aplicar_homogenea(matriz, poligono)

    det_lineal = float(np.linalg.det(matriz[:2, :2]))
    area_original = float(area_poligono(poligono))
    area_nueva = float(area_poligono(transformado))
    lados_original = longitudes_lados(poligono)
    lados_nuevo = longitudes_lados(transformado)

    return {
        "transformado": transformado,
        "determinante": det_lineal,
        "determinante_3x3": float(np.linalg.det(matriz)),
        "area_original": area_original,
        "area_transformada": area_nueva,
        "razon_area": area_nueva / area_original if area_original > tolerancia else np.nan,
        "conserva_area": bool(abs(area_nueva - area_original) <= tolerancia),
        "conserva_longitudes": bool(np.allclose(lados_original, lados_nuevo, atol=tolerancia)),
        "lados_original": lados_original,
        "lados_transformado": lados_nuevo,
    }


def verificar_orden_importa(matriz_a, matriz_b, tolerancia=1e-9):
    """Compara A·B con B·A. Devuelve (AB, BA, son_distintas)."""
    a = _validar_homogenea(matriz_a)
    b = _validar_homogenea(matriz_b)
    ab = a @ b
    ba = b @ a
    return ab, ba, bool(not np.allclose(ab, ba, atol=tolerancia))


# ----------------------------------------------------------------------
# Visualización
# ----------------------------------------------------------------------

def graficar_poligono_transformado(matriz, poligono, titulo="Transformación de un polígono",
                                   etiquetas=None, guardar_en=None, mostrar=True):
    """Dibuja el polígono original y el transformado con ejes, leyenda y vértices
    rotulados. Devuelve el diccionario de `analizar_transformacion`."""
    import matplotlib.pyplot as plt

    poligono = _validar_poligono(poligono)
    analisis = analizar_transformacion(matriz, poligono)
    nuevo = analisis["transformado"]

    if etiquetas is None:
        etiquetas = [chr(65 + i) for i in range(len(poligono))]

    if len(etiquetas) != len(poligono):
        raise ValueError("Debe haber una etiqueta por vértice.")

    fig, ax = plt.subplots(figsize=(7, 7))

    cerrado = np.vstack((poligono, poligono[0]))
    cerrado_n = np.vstack((nuevo, nuevo[0]))

    ax.fill(cerrado[:, 0], cerrado[:, 1], alpha=0.25, color="tab:blue")
    ax.plot(cerrado[:, 0], cerrado[:, 1], "o-", color="tab:blue",
            label=f"Original (área = {analisis['area_original']:.2f})")
    ax.fill(cerrado_n[:, 0], cerrado_n[:, 1], alpha=0.25, color="tab:red")
    ax.plot(cerrado_n[:, 0], cerrado_n[:, 1], "s-", color="tab:red",
            label=f"Transformada (área = {analisis['area_transformada']:.2f})")

    for etiqueta, (x, y) in zip(etiquetas, poligono):
        ax.annotate(etiqueta, (x, y), textcoords="offset points", xytext=(6, 6), color="tab:blue")

    for etiqueta, (x, y) in zip(etiquetas, nuevo):
        ax.annotate(f"{etiqueta}'", (x, y), textcoords="offset points", xytext=(6, 6), color="tab:red")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":")
    ax.set_xlabel("Eje x")
    ax.set_ylabel("Eje y")
    ax.set_title(f"{titulo}\ndet = {analisis['determinante']:.2f}")
    ax.legend(loc="best")

    if guardar_en:
        fig.savefig(guardar_en, dpi=200, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

    return analisis