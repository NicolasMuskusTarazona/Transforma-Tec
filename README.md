# Transforma-Tec: El Gran Reto Final

Proyecto académico de Álgebra Lineal de la Universidad de Santander (UDES), programa de Ingeniería de Software.

**Integrantes:** Nicolás Muskus Tarazona y Danilo Muskus Tarazona  
**Docente:** Islenis Carolina Botello Cuvides

## 1. Descripción

Transforma-Tec integra módulos de Python para operaciones con vectores y matrices, sistemas lineales mediante eliminación de Gauss, transformaciones 2D, espacios vectoriales y gamificación progresiva.

## 2. Estructura del proyecto

```text
Transforma-Tec/
├── src/                         # Módulos reutilizables
├── geogebra/                    # Instrucciones de GeoGebra
├── informe/                     # Material de apoyo para anexos
├── test_*.py                    # Pruebas automatizadas
├── celdas_colab_sistemas.py     # Fuente de las celdas ampliadas de Gauss
├── README.md
└── Transforma_Tec_Reto_Final.ipynb
```

## 3. Requisitos

Python 3, NumPy, Matplotlib, Pandas e ipywidgets. En Google Colab normalmente NumPy y Matplotlib ya están disponibles. Si hace falta, ejecuta:

```python
%pip install -q numpy matplotlib pandas ipywidgets
```

## 4. Ejecución en Google Colab (instrucción única)

1. Descarga y sube a la carpeta `/content` el archivo **`Transforma-Tec_Colab_actualizado.zip`**.
2. Extrae el ZIP en `/content` (debe quedar `/content/Transforma-Tec/`).
3. Abre `Transforma_Tec_Reto_Final.ipynb` con Google Colab.
4. Selecciona **Entorno de ejecución → Ejecutar todas** y espera a que terminen las celdas. La primera celda detecta la carpeta del proyecto.
5. Comprueba que la celda de pruebas finalice con el mensaje de que todas las pruebas finalizaron correctamente.
6. En la sección **6. Gamificación**, ejecuta la celda **DEMO INTERACTIVA** y responde los retos mediante el control de la interfaz.
7. Al finalizar, usa **Archivo → Guardar** o `Ctrl+S` para conservar el notebook con sus salidas.
8. Para entregar el cuaderno, comparte el enlace de Colab con permiso de lector y conserva también el archivo `.ipynb` descargado.

Si aparece `ModuleNotFoundError`, verifica que `/content/Transforma-Tec/src/` exista y vuelve a ejecutar la primera celda. Si los controles de ipywidgets no aparecen, ejecuta la instalación de requisitos y vuelve a ejecutar la celda de demo.

## 5. Ejecución local de pruebas

Desde una terminal ubicada en la carpeta `Transforma-Tec`:

```bash
python -m pip install numpy matplotlib pandas ipywidgets
python test_vectores.py
python test_matrices.py
python test_sistemas.py
python test_transformaciones.py
python test_espacios_vectoriales.py
python test_gamificacion.py
python test_aplicaciones.py
```

## 6. Contenido ampliado del módulo de Gauss

La sección 3 del notebook incluye pruebas de sistemas 2×2, 3×3 y 4×4, homogéneos y no homogéneos, clasificación por rangos, parametrización de sistemas con infinitas soluciones y contraste con `numpy.linalg.solve`. También presenta una aplicación de balance de carga entre servidores y un modelo de circulación de tráfico en una red. El archivo `celdas_colab_sistemas.py` se conserva como fuente de esas celdas integradas.

## 7. Mecánica de gamificación

El módulo `src/gamificacion.py` implementa cinco niveles con tres retos por nivel, selección reproducible con semilla `2026`, tres vidas iniciales, diez puntos por respuesta correcta, insignias por nivel y desbloqueo al obtener al menos dos respuestas correctas de tres. La partida finaliza al completar los cinco niveles o al agotarse las vidas.

## 8. Código fuente y evidencias

Los módulos `.py` se encuentran en `src/`, junto con los archivos `test_*.py` y `celdas_colab_sistemas.py`. Guarda las capturas reales de ejecución en una carpeta `evidencia/` cuando las generes. Las instrucciones de GeoGebra están en `geogebra/comandos_geogebra.txt`; la corrección de GeoGebra se realizará en una etapa separada.
