# TP Algoritmos Genéticos - Inteligencia Artificial

Cátedra Inteligencia Artificial, Lic. en Sistemas de Información (FCyT, UADER, Oro Verde), 2026.

Integrantes: Facundo Tomas Hernandez y Facundo Thomas Ramirez

Resolución de la Parte 1 del TP (Algoritmos Genéticos). Todo está en el notebook
`TP_AG_Hernandez_Ramirez.ipynb`, separado por actividad y, dentro de cada una, por etapa
del AG (población, aptitud, selección, cruzamiento y mutación).

- Actividad 1: búsqueda de la cadena "Los algoritmos genéticos sirven para optimizar" y elección
  automática del mejor tamaño de población de PS = [100, 1000, 50, 200, 300, 50, 500].
- Actividad 2.a: mínimo de f(x,y) = (x-1)^2 + (x-y)^4 con selección por ruleta.
- Actividad 2.b: máximo de f(x,y) = 2 + sqrt(9 - x^2 - y^2) con selección por rango.
- Actividad 2.c: máximo de f(x,y,z) = z - x^2 - 15y^2 con -2 < x,y,z < 11, con selección por torneo.
- Actividad 3: selección de variables en el dataset Wine.

## Archivos

- `TP_AG_Hernandez_Ramirez.ipynb`: el entregable, con el código, las explicaciones, las salidas y los gráficos.
- `TP_AG_Hernandez_Ramirez.py`: el mismo código exportado como script, para leerlo o correrlo fuera de Jupyter.

## Cómo ejecutarlo

```
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook TP_AG_Hernandez_Ramirez.ipynb
```

El notebook ya tiene las salidas guardadas. Correrlo entero tarda alrededor de un minuto.
