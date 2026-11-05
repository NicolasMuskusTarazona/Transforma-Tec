# %% [markdown]
# ## Celda A (Markdown) — Sistemas homogéneos y no homogéneos
#
# Un sistema lineal $A\vec{x}=\vec{b}$ es **homogéneo** cuando $\vec{b}=\vec{0}$ y **no homogéneo** cuando $\vec{b}\neq\vec{0}$. Un sistema homogéneo siempre es compatible porque admite la solución trivial $\vec{x}=\vec{0}$; tiene infinitas soluciones únicamente si $\operatorname{rango}(A)<n$.
#
# La función `eliminacion_gauss` clasifica cada caso con los rangos de $A$ y de la matriz ampliada $[A\,|\,\vec{b}]$ (teorema de Rouché–Frobenius) y la función `comparar_con_numpy` contrasta el resultado con `numpy.linalg.solve`.

# %% [code]
# Celda B — Batería de pruebas: 2x2, 3x3 y 4x4, homogéneos y no homogéneos
import numpy as np
import matplotlib.pyplot as plt
from src.sistemas import eliminacion_gauss, comparar_con_numpy

casos = [
    ("2x2 no homogéneo", [[2, 1], [1, -1]], [5, 1]),
    ("2x2 homogéneo", [[1, 2], [3, 4]], [0, 0]),
    ("3x3 no homogéneo", [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3]),
    ("3x3 homogéneo (trivial)", [[1, 1, 1], [1, 2, 3], [2, 1, 4]], [0, 0, 0]),
    ("3x3 homogéneo (no trivial)", [[1, 2, 3], [2, 4, 6], [1, 0, 1]], [0, 0, 0]),
    ("3x3 infinitas soluciones", [[1, 1, 1], [1, 2, 3], [2, 3, 4]], [6, 14, 20]),
    ("4x4 no homogéneo", [[1, 1, 1, 1], [2, -1, 0, 3], [0, 1, -1, 2], [3, 0, 2, -1]], [10, 9, 3, 7]),
    ("4x4 homogéneo (no trivial)", [[1, 1, 0, 1], [0, 1, 1, 0], [1, 2, 1, 1], [2, 3, 1, 2]], [0, 0, 0, 0]),
    ("4x4 incompatible", [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 2, 2, 1]], [1, 2, 3, 20]),
]

print(f"{'Caso':<28}{'Tipo':<14}{'rg(A)':<7}{'rg(A|b)':<9}{'Clasificación':<26}{'Verif. NumPy'}")
print("-" * 100)
for nombre, A, b in casos:
    r = eliminacion_gauss(A, b)
    cmp = comparar_con_numpy(A, b)
    tipo = "homogéneo" if r.es_homogeneo else "no homogéneo"
    clase = r.clasificacion.split(" (")[0]
    verif = "OK" if cmp["coincide"] else "FALLA"
    if cmp["metodo"] == "numpy.linalg.solve":
        verif += f" (error {cmp['error_maximo']:.1e})"
    else:
        verif += " (rangos)"
    print(f"{nombre:<28}{tipo:<14}{r.rango_A:<7}{r.rango_ampliada:<9}{clase:<26}{verif}")
    assert cmp["coincide"], nombre

# %% [code]
# Celda C — Solución detallada de tres casos representativos
for nombre, A, b in [casos[2], casos[5], casos[7]]:
    print(f"=== {nombre} ===")
    print(eliminacion_gauss(A, b))
    print()

# %% [markdown]
# ## Celda D (Markdown) — Aplicación en Ingeniería de Software: balance de carga
#
# Un balanceador de carga reparte $T$ peticiones por segundo entre $n$ servidores con capacidades relativas $w_1,\dots,w_n$. Sea $x_i$ la carga asignada al servidor $i$. El modelo es:
#
# $$
# \begin{cases}
# x_1 + x_2 + \cdots + x_n = T & \text{(se atiende toda la demanda)}\\
# w_1 x_i - w_i x_1 = 0,\quad i=2,\dots,n & \text{(carga proporcional a la capacidad)}
# \end{cases}
# $$
#
# Es un sistema no homogéneo $n\times n$ cuyas $n-1$ últimas ecuaciones son homogéneas. Con $T=0$ el sistema completo pasa a ser homogéneo y solo admite la solución trivial (no hay demanda que repartir).

# %% [code]
# Celda E — Balance de carga con 3 servidores (capacidades 4, 2 y 5; T = 1100 req/s)
from src.aplicaciones import sistema_balance_carga, balancear_carga

pesos = [4, 2, 5]
total = 1100
A, b = sistema_balance_carga(total, pesos)
resultado = balancear_carga(total, pesos)

print("Matriz de coeficientes A:\n", A)
print("Vector b:", b)
print()
print(resultado)
print()
print(f"{'Servidor':<10}{'Capacidad':<11}{'Carga (req/s)':<15}{'% del total'}")
for i, (w, x) in enumerate(zip(pesos, resultado.solucion), start=1):
    print(f"S{i:<9}{w:<11}{x:<15.1f}{100 * x / total:.1f} %")

cmp = comparar_con_numpy(A, b)
print("\nVerificación contra numpy.linalg.solve:", cmp["detalle"])
assert cmp["coincide"]

fig, ax = plt.subplots(figsize=(7, 4.5))
nombres = [f"S{i}" for i in range(1, len(pesos) + 1)]
barras = ax.bar(nombres, resultado.solucion, color=["#4C72B0", "#55A868", "#C44E52"])
for barra, w in zip(barras, pesos):
    ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 8,
            f"{barra.get_height():.0f} req/s\n(capacidad {w})", ha="center", fontsize=9)
ax.set_xlabel("Servidor")
ax.set_ylabel("Carga asignada (peticiones/s)")
ax.set_title("Carga por servidor obtenida con eliminación de Gauss (T = 1100 req/s)")
ax.set_ylim(0, max(resultado.solucion) * 1.25)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [code]
# Celda F — Variantes del modelo: 4 servidores, política incompleta y requisitos contradictorios
# (a) 4 servidores (sistema 4x4)
pesos4 = [3, 2, 1, 4]
A4, b4 = sistema_balance_carga(1000, pesos4)
r4 = balancear_carga(1000, pesos4)
print("(a) 4 servidores, capacidades", pesos4)
print(r4)
print("   Verificación NumPy:", comparar_con_numpy(A4, b4)["detalle"])

# (b) Política incompleta: se omite la regla del último servidor -> infinitas soluciones
A_inc, b_inc = A[:-1], b[:-1]
r_inc = eliminacion_gauss(A_inc, b_inc)
print("\n(b) Política incompleta (3 servidores, 2 ecuaciones)")
print(r_inc)
print("   Ejemplo con t1 = 300:", r_inc.solucion_para([300.0]))

# (c) Requisitos contradictorios: se exige además que el total sea 900 req/s
A_con = np.vstack([A, np.ones(3)])
b_con = np.append(b, 900.0)
r_con = eliminacion_gauss(A_con, b_con)
print("\n(c) Requisitos contradictorios (total = 1100 y total = 900)")
print(r_con)

# (d) Sin demanda (sistema homogéneo): solo la solución trivial
print("\n(d) Sin demanda (T = 0)")
print(balancear_carga(0, pesos))

# %% [markdown]
# ## Celda G (Markdown) — Sistema homogéneo aplicado: circulación de tráfico en una red
#
# En una red de tres enrutadores sin tráfico externo, el tráfico que entra a cada nodo debe ser igual al que sale. Con los enlaces $f_1:R_1\to R_2$, $f_2:R_2\to R_3$, $f_3:R_3\to R_1$ y $f_4:R_1\to R_3$ se obtiene un sistema **homogéneo** de rango 2 con cuatro incógnitas, por lo que existen infinitas circulaciones posibles:
#
# $$
# \begin{cases}
# f_3 - f_1 - f_4 = 0 & (R_1)\\
# f_1 - f_2 = 0 & (R_2)\\
# f_2 + f_4 - f_3 = 0 & (R_3)
# \end{cases}
# $$

# %% [code]
# Celda H — Resolución de la red de enrutadores
from src.aplicaciones import sistema_flujo_red, analizar_flujo_red

A_red, b_red = sistema_flujo_red()
r_red = analizar_flujo_red()
print("Matriz de conservación A:\n", A_red)
print()
print(r_red)
print()
print("Ejemplo físico (t1 = 5, t2 = 2) -> f =", r_red.solucion_para([5, 2]))
print("Comprobación A·f = 0:", np.allclose(A_red @ r_red.solucion_para([5, 2]), 0))
print("Verificación NumPy:", comparar_con_numpy(A_red, b_red)["detalle"])
print("\nNota: para que los flujos sean físicamente válidos (f ≥ 0) se requiere t1 ≥ t2 ≥ 0.")