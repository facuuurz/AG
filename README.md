# TP Algoritmos Genéticos — Inteligencia Artificial

**Cátedra:** Inteligencia Artificial — Lic. en Sistemas de Información (FCyT, UADER, Oro Verde) — 2026
**Integrantes:** Facundo Tomas Hernandez · Facundo Thomas Ramirez

Resolución de la Parte 1 del TP (Algoritmos Genéticos) en el notebook
[`TP_AG_Hernandez_Ramirez.ipynb`](TP_AG_Hernandez_Ramirez.ipynb), con el código separado
por etapa según el *proceso usual AG*: población → aptitud → selección → cruzamiento → mutación.

## Contenido

| Actividad | Problema | Codificación | Selección |
|---|---|---|---|
| 1 | Búsqueda de la cadena `"Los algoritmos genéticos sirven para optimizar"` y elección automática del mejor tamaño de población de `PS = [100, 1000, 50, 200, 300, 50, 500]` | Entera | Torneo + elitismo |
| 2.a | Mínimo de `f(x,y) = (x-1)² + (x-y)⁴` | Binaria | Rueda de ruleta |
| 2.b | Máximo de `f(x,y) = 2 + sqrt(9 - x² - y²)` (parte real) | Real | Rango lineal + penalización |
| 2.c | Máximo de `f(x,y,z) = z - x² - 15y²` con `-2 < x,y,z < 11` | Real | Torneo + elitismo |
| 3 | Selección de variables en el dataset *Wine* (etapas Seleccionar / Cruzar / Mutar / Reemplazar) | Binaria | Torneo |

## Archivos

- `TP_AG_Hernandez_Ramirez.ipynb` — **el entregable**: notebook con código Python, explicaciones, salidas y gráficos.
- `TP_AG_Hernandez_Ramirez.py` — el mismo código Python exportado como script plano (sin las explicaciones ni los gráficos embebidos), por si se quiere leer o correr fuera de Jupyter.

## Ejecución

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook TP_AG_Hernandez_Ramirez.ipynb
```

El notebook ya incluye todas las salidas y gráficos. Re-ejecutarlo completo tarda unos 2 minutos
(Python 3.10, NumPy 2.x, scikit-learn 1.7).
