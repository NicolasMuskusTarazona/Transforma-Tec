"""Aplicaciones de sistemas lineales en Ingeniería de Software.

Casos modelados
---------------
1. Balance de carga entre servidores proporcional a su capacidad
   (sistema no homogéneo con ecuaciones de proporcionalidad homogéneas).
2. Conservación del tráfico en una red de enrutadores sin entradas ni salidas
   externas (sistema homogéneo: circulación de flujo).
"""

import numpy as np

from src.sistemas import eliminacion_gauss, validar_sistema


def _validar_pesos(pesos):
    pesos = np.asarray(pesos, dtype=float)
    if pesos.ndim != 1 or pesos.size < 2:
        raise ValueError("Se requieren al menos dos servidores (vector de pesos).")
    if not np.all(np.isfinite(pesos)) or np.any(pesos <= 0):
        raise ValueError("Las capacidades (pesos) deben ser números positivos.")
    return pesos


def sistema_balance_carga(total, pesos):
    """Construye el sistema Ax = b del balance de carga proporcional.

    Incógnitas: x_i = peticiones por segundo asignadas al servidor i.

    Ecuaciones:
      * Conservación:     x_1 + x_2 + ... + x_n = total
      * Proporcionalidad: w_1·x_i - w_i·x_1 = 0   (i = 2..n)
        (el servidor i recibe carga en proporción a su capacidad w_i)

    Si total = 0 el sistema es homogéneo.
    """
    pesos = _validar_pesos(pesos)
    total = float(total)
    if not np.isfinite(total) or total < 0:
        raise ValueError("La carga total debe ser un número no negativo.")

    n = pesos.size
    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0, :] = 1.0
    b[0] = total
    for i in range(1, n):
        A[i, 0] = -pesos[i]
        A[i, i] = pesos[0]

    return A, b


def balancear_carga(total, pesos):
    """Resuelve el balance de carga y devuelve el ResultadoSistema."""
    A, b = sistema_balance_carga(total, pesos)
    return eliminacion_gauss(A, b)


def sistema_flujo_red():
    """Red de 3 enrutadores con 4 enlaces dirigidos (conservación de tráfico).

    Enlaces:  f1: R1→R2,  f2: R2→R3,  f3: R3→R1,  f4: R1→R3.
    Sin tráfico externo, lo que entra a cada nodo es igual a lo que sale:

      R1:  f3 - f1 - f4 = 0
      R2:  f1 - f2      = 0
      R3:  f2 + f4 - f3 = 0

    Es un sistema homogéneo con rango 2 y 4 incógnitas.
    """
    A = np.array(
        [
            [-1.0, 0.0, 1.0, -1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 1.0],
        ]
    )
    b = np.zeros(3)
    validar_sistema(A, b)
    return A, b


def analizar_flujo_red():
    """Resuelve la red de enrutadores y devuelve el ResultadoSistema."""
    A, b = sistema_flujo_red()
    return eliminacion_gauss(A, b)