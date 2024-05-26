import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# To-do:
# - cuadrados mínimos.
# - gráfico sobre como se distribuyen las muestras? Estan en nxp dimensiones, cómo hago ***


def distancia_euclidiana(xi, xj, sigma):
    
    norma_2 = np.sum((xi - xj) ** 2) # Sin la raíz porque lo compenso en la cuenta de abajo.
    
    res = np.exp( - norma_2 / (2 * sigma**2))
    
    return res

def graficar_matriz_similitud(similarity_matrix, d):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title(f"Matriz de Similitud. d = {d}.")
    plt.xlabel("Índice de Muestra")
    plt.ylabel("Índice de Muestra")
    plt.show()

# Desempaquetado de los datos en el archivo csv, guardados en una matriz
data_matrix_X = np.loadtxt('dataset03.csv', delimiter=",", skiprows=1)

# Descomposición SVD:
U, S, Vt = np.linalg.svd(data_matrix_X, full_matrices=False) 
S = np.diag(S) #svd me devuelve un vector de valores singulares, lo convierto a matriz.

# PCA: matriz de similaridad par a par entre cada fila

# Reducir la dimensionalidad: Matrices de rango d más cercanas a X
for d in (2,6,10, data_matrix_X.shape[0]):
    print("Matriz reducida. d = ", d, ".")
    X_aprox_d= U[:,:d]@S[:d,:d]@Vt[:d,:]

    #Modelo de PCA a utilizar:
    pca = PCA(n_components=d)
    X_pca_d = pca.fit_transform(data_matrix_X)

    # Inicializar la matriz de similitud
    n = X_pca_d.shape[0]
    similarity_matrix = np.zeros((n, n))
    
    # Calcular la similitud par a par
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i, j] = distancia_euclidiana(X_pca_d[i], X_pca_d[j], 1.0) # sigma??? ***
            else:
                similarity_matrix[i, j] = 1.0  # La similitud de una fila consigo misma es 1

    graficar_matriz_similitud(similarity_matrix, d)
    


    


