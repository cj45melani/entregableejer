import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print("\n--- Ejercicio 1: Producto punto ---")
print("Producto punto:", v1 @ v2)

v = np.array([1, 2, 3])
print("\n--- Ejercicio 2: Norma y escalar ---")
print("Norma de v:", np.linalg.norm(v))
print("v por 5:", v * 5)

A = np.array([[2, 1],
              [1, -1]])
b = np.array([5, 1])
sol = np.linalg.solve(A, b)
print("\n--- Ejercicio 3: Sistema de ecuaciones ---")
print("Solución del sistema:", sol)

datos = pd.Series([2, 4, 6, 8, 50])
print("\n--- Ejercicio 4: Media y mediana ---")
print("Media:", datos.mean())
print("Mediana:", datos.median())

m = np.mean(datos)
s = np.std(datos)
outliers = [x for x in datos if abs(x - m) > 2 * s]
print("\n--- Ejercicio 5: Outliers ---")
print("Outliers:", outliers)

# Histograma
plt.figure(figsize=(6,4))
plt.hist(datos, bins=5, color='red', edgecolor='black')
plt.title('Histograma de datos')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de dispersión
df = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 3, 5, 7, 11],
    'Categoría': ['A','B','A','B','A']
})
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='X', y='Y', hue='Categoría', s=100)
plt.title('Gráfico de dispersión con Seaborn')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.show()
