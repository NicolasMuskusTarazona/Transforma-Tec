"""Módulo 3 – Sistemas de ecuaciones lineales.

Implementa la eliminación de Gauss con pivoteo parcial y tolerancia numérica
para sistemas Ax = b de cualquier tamaño (m ecuaciones, n incógnitas). Una
sola llamada a :func:`eliminacion_gauss` realiza la eliminación hacia adelante,
clasifica el sistema y, según el caso, devuelve:

* solución única (compatible determinado),
* parametrización de la solución general (compatible indeterminado), o
* la constancia de que no existe solución (incompatible).
"""

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

COMPATIBLE_DETERMINADO = "Compatible determinado (solución única)"
COMPATIBLE_INDETERMINADO = "Compatible indeterminado (infinitas soluciones)"
INCOMPATIBLE = "Incompatible (sin solución)"


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def validar_sistema(A, b):
    """Valida la matriz de coeficientes y el vector de términos independientes.

    Parámetros
    ----------
    A : array_like de forma (m, n)
    b : array_like de forma (m,)

    Devuelve
    --------
    (A, b) como arreglos de NumPy de tipo float.

    Lanza
    -----
    ValueError si las dimensiones no son coherentes, el sistema está vacío o
    contiene valores no numéricos o no finitos.
    """
    try:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "A y b deben contener únicamente valores numéricos "
            "y tener filas de igual longitud."
        ) from error

    if A.ndim != 2:
        raise ValueError(
            f"A debe ser una matriz (2 dimensiones); se recibió {A.ndim}."
        )

    if b.ndim != 1:
        raise ValueError(
            f"b debe ser un vector (1 dimensión); se recibió {b.ndim}."
        )

    if A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError("El sistema no puede estar vacío.")

    if A.shape[0] != b.size:
        raise ValueError(
            f"A tiene {A.shape[0]} ecuaciones pero b tiene {b.size} elementos."
        )

    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("El sistema contiene valores no finitos (NaN o inf).")

    return A, b


def _validar_tolerancia(tolerancia):
    if not np.isscalar(tolerancia) or not tolerancia > 0:
        raise ValueError("La tolerancia debe ser un número positivo.")


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
def _numero(valor):
    """Da formato legible (fracción exacta si es posible) a un número."""
    valor = float(valor)
    fraccion = Fraction(valor).limit_denominator(1000)
    if abs(float(fraccion) - valor) < 1e-9:
        return str(fraccion)
    return f"{valor:.4g}"


def _expresion(constante, coeficientes, nombres):
    """Construye texto como '3 - 2·t1 + 1/2·t2'."""
    partes = []
    if abs(constante) > 1e-12 or not any(abs(c) > 1e-12 for c in coeficientes):
        partes.append(_numero(constante))

    for coef, nombre in zip(coeficientes, nombres):
        if abs(coef) <= 1e-12:
            continue
        signo = "-" if coef < 0 else "+"
        magnitud = abs(coef)
        termino = nombre if np.isclose(magnitud, 1) else f"{_numero(magnitud)}·{nombre}"
        if not partes:
            partes.append(f"-{termino}" if signo == "-" else termino)
        else:
            partes.append(f"{signo} {termino}")

    return " ".join(partes)


@dataclass
class ResultadoSistema:
    """Resultado completo de la eliminación de Gauss.

    Atributos
    ---------
    clasificacion : str
        Una de COMPATIBLE_DETERMINADO, COMPATIBLE_INDETERMINADO, INCOMPATIBLE.
    rango_A, rango_ampliada : int
        Rangos de la matriz de coeficientes y de la matriz ampliada.
    es_homogeneo : bool
        True si b = 0.
    matriz_escalonada : ndarray
        Matriz ampliada [A | b] tras la eliminación hacia adelante.
    pivotes : list[int]
        Índices (base 0) de las columnas pivote.
    variables_libres : list[int]
        Índices (base 0) de las incógnitas libres (parámetros).
    solucion : ndarray | None
        Solución única; None si el sistema no es compatible determinado.
    particular : ndarray | None
        Solución particular (variables libres = 0) si hay infinitas soluciones.
    direcciones : ndarray | None
        Fila k = vector director asociado al parámetro t(k+1).
    """

    clasificacion: str
    rango_A: int
    rango_ampliada: int
    es_homogeneo: bool
    matriz_escalonada: np.ndarray
    pivotes: list = field(default_factory=list)
    variables_libres: list = field(default_factory=list)
    solucion: np.ndarray = None
    particular: np.ndarray = None
    direcciones: np.ndarray = None

    @property
    def n_incognitas(self):
        return self.matriz_escalonada.shape[1] - 1

    @property
    def grados_libertad(self):
        return len(self.variables_libres) if self.clasificacion != INCOMPATIBLE else 0

    def solucion_para(self, parametros):
        """Evalúa la solución general x = x_p + Σ t_k·d_k para valores dados."""
        if self.clasificacion != COMPATIBLE_INDETERMINADO:
            raise ValueError("Solo hay parametrización con infinitas soluciones.")
        parametros = np.asarray(parametros, dtype=float)
        if parametros.shape != (len(self.variables_libres),):
            raise ValueError(
                f"Se esperaban {len(self.variables_libres)} parámetros."
            )
        return self.particular + parametros @ self.direcciones

    def parametrizacion(self):
        """Devuelve la solución general como lista de textos, uno por incógnita."""
        if self.clasificacion != COMPATIBLE_INDETERMINADO:
            raise ValueError("Solo hay parametrización con infinitas soluciones.")

        parametros = [f"t{k + 1}" for k in range(len(self.variables_libres))]
        lineas = []
        for i in range(self.n_incognitas):
            if i in self.variables_libres:
                k = self.variables_libres.index(i)
                lineas.append(f"x{i + 1} = {parametros[k]}  (variable libre)")
            else:
                expr = _expresion(
                    self.particular[i], self.direcciones[:, i], parametros
                )
                lineas.append(f"x{i + 1} = {expr}")
        return lineas

    def resumen(self):
        """Texto legible con la clasificación y la solución."""
        texto = [
            f"Clasificación: {self.clasificacion}",
            f"rango(A) = {self.rango_A}, rango([A|b]) = {self.rango_ampliada}, "
            f"incógnitas = {self.n_incognitas}",
            f"Tipo: {'homogéneo' if self.es_homogeneo else 'no homogéneo'}",
        ]
        if self.clasificacion == COMPATIBLE_DETERMINADO:
            for i, valor in enumerate(self.solucion):
                texto.append(f"x{i + 1} = {_numero(valor)}")
        elif self.clasificacion == COMPATIBLE_INDETERMINADO:
            texto.append(
                f"Solución general ({len(self.variables_libres)} parámetro(s) libre(s)):"
            )
            texto.extend(self.parametrizacion())
        else:
            texto.append("No existe ningún vector x que satisfaga Ax = b.")
        return "\n".join(texto)

    def __str__(self):
        return self.resumen()


# ---------------------------------------------------------------------------
# Eliminación de Gauss
# ---------------------------------------------------------------------------
def _sustitucion_hacia_atras(matriz, pivotes, n, libres, valores_libres, usar_b):
    """Sustitución hacia atrás fijando las variables libres.

    Si ``usar_b`` es False se resuelve el sistema homogéneo asociado, lo que
    produce los vectores directores de la solución general.
    """
    x = np.zeros(n)
    x[libres] = valores_libres

    for fila in range(len(pivotes) - 1, -1, -1):
        columna = pivotes[fila]
        suma = matriz[fila, columna + 1:n] @ x[columna + 1:n]
        termino = matriz[fila, -1] if usar_b else 0.0
        x[columna] = (termino - suma) / matriz[fila, columna]

    return x


def eliminacion_gauss(A, b, tolerancia=1e-10):
    """Resuelve y clasifica Ax = b mediante eliminación de Gauss.

    Usa pivoteo parcial (mayor valor absoluto de la columna) y una tolerancia
    relativa a la magnitud de los datos para decidir cuándo un pivote es cero.
    Funciona con sistemas cuadrados, sobredeterminados o subdeterminados.

    Parámetros
    ----------
    A : array_like (m, n)   matriz de coeficientes.
    b : array_like (m,)     términos independientes.
    tolerancia : float      umbral para considerar un valor igual a cero.

    Devuelve
    --------
    ResultadoSistema
    """
    A, b = validar_sistema(A, b)
    _validar_tolerancia(tolerancia)

    m, n = A.shape
    matriz = np.column_stack((A, b))
    umbral = tolerancia * max(1.0, np.max(np.abs(matriz)))

    # Eliminación hacia adelante con pivoteo parcial.
    pivotes = []
    fila = 0
    for columna in range(n):
        if fila >= m:
            break

        candidato = fila + int(np.argmax(np.abs(matriz[fila:, columna])))
        if abs(matriz[candidato, columna]) <= umbral:
            matriz[fila:, columna] = 0.0  # columna sin pivote: variable libre
            continue

        matriz[[fila, candidato]] = matriz[[candidato, fila]]

        for j in range(fila + 1, m):
            factor = matriz[j, columna] / matriz[fila, columna]
            matriz[j, columna:] -= factor * matriz[fila, columna:]
            matriz[j, columna] = 0.0

        pivotes.append(columna)
        fila += 1

    rango_A = len(pivotes)
    libres = [c for c in range(n) if c not in pivotes]
    es_homogeneo = bool(np.all(np.abs(b) <= umbral))

    # Filas sin pivote con término independiente distinto de cero: 0 = c, c ≠ 0.
    inconsistente = bool(np.any(np.abs(matriz[rango_A:, -1]) > umbral))

    if inconsistente:
        return ResultadoSistema(
            clasificacion=INCOMPATIBLE,
            rango_A=rango_A,
            rango_ampliada=rango_A + 1,
            es_homogeneo=es_homogeneo,
            matriz_escalonada=matriz,
            pivotes=pivotes,
            variables_libres=libres,
        )

    # Sistema compatible: sustitución hacia atrás.
    particular = _sustitucion_hacia_atras(
        matriz, pivotes, n, libres, np.zeros(len(libres)), usar_b=True
    )

    if not libres:
        return ResultadoSistema(
            clasificacion=COMPATIBLE_DETERMINADO,
            rango_A=rango_A,
            rango_ampliada=rango_A,
            es_homogeneo=es_homogeneo,
            matriz_escalonada=matriz,
            pivotes=pivotes,
            variables_libres=libres,
            solucion=particular,
        )

    direcciones = np.array(
        [
            _sustitucion_hacia_atras(
                matriz, pivotes, n, libres, np.eye(len(libres))[k], usar_b=False
            )
            for k in range(len(libres))
        ]
    )

    return ResultadoSistema(
        clasificacion=COMPATIBLE_INDETERMINADO,
        rango_A=rango_A,
        rango_ampliada=rango_A,
        es_homogeneo=es_homogeneo,
        matriz_escalonada=matriz,
        pivotes=pivotes,
        variables_libres=libres,
        particular=particular,
        direcciones=direcciones,
    )


def clasificar_sistema(A, b, tolerancia=1e-10):
    """Devuelve la clasificación del sistema (texto) usando Gauss."""
    return eliminacion_gauss(A, b, tolerancia).clasificacion


def resolver_sistema(A, b, tolerancia=1e-10):
    """Resuelve Ax = b y devuelve un :class:`ResultadoSistema`.

    A diferencia de versiones anteriores, no lanza error cuando hay infinitas
    soluciones ni cuando el sistema es incompatible: el resultado describe cada
    caso (solución, parametrización o ausencia de solución).
    """
    return eliminacion_gauss(A, b, tolerancia)


def comparar_con_numpy(A, b, tolerancia=1e-10):
    """Contrasta la eliminación de Gauss propia con NumPy.

    * Si el sistema es cuadrado y de rango completo, compara la solución con
      ``numpy.linalg.solve`` y reporta el error máximo.
    * En los demás casos ``numpy.linalg.solve`` no es aplicable (matriz
      singular o no cuadrada); entonces se comprueba que el rango calculado
      coincida con ``numpy.linalg.matrix_rank`` y que la solución (única o
      particular, si existe) satisfaga Ax = b.

    Devuelve un diccionario con las claves: ``coincide`` (bool),
    ``metodo`` (str), ``error_maximo`` (float o None) y ``detalle`` (str).
    """
    A, b = validar_sistema(A, b)
    resultado = eliminacion_gauss(A, b, tolerancia)
    m, n = A.shape

    rango_numpy = int(np.linalg.matrix_rank(A))
    rango_ok = rango_numpy == resultado.rango_A

    if m == n and rango_numpy == n:
        referencia = np.linalg.solve(A, b)
        error = float(np.max(np.abs(resultado.solucion - referencia)))
        coincide = bool(rango_ok and error < 1e-8)
        return {
            "coincide": coincide,
            "metodo": "numpy.linalg.solve",
            "error_maximo": error,
            "detalle": f"Error máximo |x_Gauss - x_NumPy| = {error:.2e}",
        }

    if resultado.clasificacion == INCOMPATIBLE:
        coincide = rango_ok
        detalle = (
            "numpy.linalg.solve no aplica (matriz singular o no cuadrada); "
            f"rango(A) coincide con NumPy: {rango_ok}. Sin solución."
        )
        error = None
    else:
        candidata = (
            resultado.solucion
            if resultado.clasificacion == COMPATIBLE_DETERMINADO
            else resultado.particular
        )
        residuo = float(np.max(np.abs(A @ candidata - b)))
        coincide = bool(rango_ok and residuo < 1e-8)
        detalle = (
            "numpy.linalg.solve no aplica (matriz singular o no cuadrada); "
            f"rango(A) coincide con NumPy: {rango_ok}; "
            f"residuo de la solución = {residuo:.2e}"
        )
        error = residuo

    return {
        "coincide": coincide,
        "metodo": "numpy.linalg.matrix_rank",
        "error_maximo": error,
        "detalle": detalle,
    }


def verificar_solucion(A, b, x, tolerancia=1e-8):
    """Comprueba si Ax = b."""
    A, b = validar_sistema(A, b)
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or x.size != A.shape[1]:
        raise ValueError("La solución tiene dimensiones incorrectas.")

    return bool(np.allclose(A @ x, b, atol=tolerancia))


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