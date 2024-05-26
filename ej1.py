import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# To-do:
# - PCA: matriz de similaridad par a par entre cada fila de X y de sus versiones reducidas.
# - cuadrados mínimos.
# - gráfico sobre como se distribuyen las muestras? Estan en nxp dimensiones, cómo hago ***

def distancia_euclidiana(xi, xj, sigma):
    
    norma_2 = np.sum((xi - xj) ** 2) # Sin la raíz porque lo compenso en la cuenta de abajo.
    
    res = np.exp( - norma_2 / (2 * sigma**2))
    
    return res

# Desempaquetado de los datos en el archivo csv, guardados en una matriz
data_matrix_X = np.loadtxt('dataset03.csv', delimiter=",", skiprows=1)

# Descomposición SVD:
U, S, Vt = np.linalg.svd(data_matrix_X, full_matrices=False) 
S = np.diag(S) #svd me devuelve un vector de valores singulares, lo convierto a matriz.

# PCA: matriz de similaridad par a par entre cada fila

# Reducir la dimensionalidad: Matrices de rango d más cercanas a X
for d in (2,6,10):
    print("Matriz reducida. d = ", d, ".")
    X_aprox_d= U[:,:d]@S[:d,:d]@Vt[:d,:]
    
    #PCA: matriz de similaridad par a par entre cada fila.
    


