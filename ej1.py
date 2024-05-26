import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# To-do:
# - cuadrados mínimos.
# - Gráfico sobre como se distribuyen las muestras? Estan en nxp dimensiones, cómo hago si lo máximo 
#   que podemos ver y graficar como humanos es nx3 ***
# - PCA:  la matriz de similaridad con diferentes d no muestra diferencia, no se ve claro porque son 
#   muchas muestras ***
# - Qué metodo utilizar para determinar qué dimensiones de las p son las más importantes***


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
cant_filas = data_matrix_X.shape[0]
cant_variables = data_matrix_X.shape[1]
y = np.loadtxt('y3.txt')

# Descomposición SVD:
U, S, Vt = np.linalg.svd(data_matrix_X, full_matrices=False) 
S = np.diag(S) #svd me devuelve un vector de valores singulares, lo convierto a matriz.

# PCA: matriz de similaridad par a par entre cada fila

# Reducir la dimensionalidad: Matrices de rango d más cercanas a X.
# Va cambiando las cantidades de variables medidas que va tomando en cuenta (FEATURES).
# Como cambio la cantidad de mediciones que tomo en cuenta? *** 
for d in (2,6,10, data_matrix_X.shape[0]):
    print("Matriz reducida. d = ", d, ".")
    X_aprox_d= U[:,:d]@S[:d,:d]@Vt[:d,:]
    print("el shape de X_aprox_d es: ", X_aprox_d.shape)
    
    #Modelo de PCA a utilizar:
    if (d > cant_variables):
        pca = PCA(n_components= cant_variables) #Te tira un error sino ***
    else:
        pca = PCA(n_components= d)
    X_pca_d = pca.fit_transform(X_aprox_d)
    print("el shape de X_pca_d es: ", X_pca_d.shape)

    # Inicializar la matriz de similitud
    n = X_pca_d.shape[0]
    print("n es: ", n)
    similarity_matrix = np.zeros((n, n))
    
    # Calcular la similitud par a par
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i, j] = distancia_euclidiana(X_pca_d[i], X_pca_d[j], 1.0) # sigma??? ***
            else:
                similarity_matrix[i, j] = 1.0  # La similitud de una fila consigo misma es 1

    graficar_matriz_similitud(similarity_matrix, d)
    

# CUADRADOS MÍNIMOS:
    


