import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Reducir la dimensionalidad: Matrices de rango d más cercanas a X.
# Va cambiando las cantidades de variables medidas que va tomando en cuenta.
for d in (2,6,10, data_matrix_X.shape[0]):
    print("Matriz reducida. d = ", d, ".")
    X_aprox_d= U[:,:d]@S[:d,:d]@Vt[:d,:]
    print("el shape de X_aprox_d es: ", X_aprox_d.shape)
    
    # Centrar la data (libro de ***):
    X_aprox_d = preprocessing.scale(X_aprox_d)

    # Modelo de PCA a utilizar:
    if (d > cant_variables):
        pca = PCA(n_components= cant_variables) #Te tira un error sino ***
    else:
        pca = PCA(n_components= d)
    X_pca_d = pca.fit_transform(X_aprox_d)
    print("el shape de X_pca_d es: ", X_pca_d.shape)
    
    # Scree plot:
    #The following code constructs the Scree plot
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()
    
    #the following code makes a fancy looking plot using PC1 and PC2
    pca_df = pd.DataFrame(X_pca_d, index=[*wt, *ko], columns=labels)
    
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('My PCA Graph')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    
    plt.show()













    # Inicializar la matriz de similitud
    n = X_pca_d.shape[0]
    print("n es: ", n)
    similarity_matrix = np.zeros((n, n))
    
    # Calcular la similitud par a par
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i, j] = distancia_euclidiana(X_pca_d[i], X_pca_d[j], 1.0) # sigma??? de momento le pongo = 1 ***
            else:
                similarity_matrix[i, j] = 1.0 
    
    # Cuadrados mínimos para este d:
    aprox_vect = LinearRegression().fit(X_pca_d, y)

    graficar_matriz_similitud(similarity_matrix, d)
    


    


