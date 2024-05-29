import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
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

# Convertir la matriz de distancia en matriz de similitud usando afinidad gaussiana
def gaussian_similarity(dist_matrix, sigma=1.0):
    return np.exp(-dist_matrix**2 / (2 * sigma**2))

# Desempaquetado de los datos en el archivo csv, guardados en una matriz
data_matrix_X = np.loadtxt('dataset03.csv', delimiter=",", skiprows=1, usecols=range(1, np.genfromtxt('dataset03.csv', delimiter=",", max_rows=1).size))
cant_filas = data_matrix_X.shape[0]
cant_variables = data_matrix_X.shape[1]
y = np.loadtxt('y3.txt')
print("cantidad de filas: ", cant_filas,". cantidad de columnas: ", cant_variables)

# Descomposición SVD:
U, S, Vt = np.linalg.svd(data_matrix_X, full_matrices=False) 
S = np.diag(S) #svd me devuelve un vector de valores singulares, lo convierto a matriz.

# Plot U, S, Vt
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot U
axes[0].imshow(U, aspect='auto', cmap='viridis')
axes[0].set_title('Matrix U')
axes[0].set_xlabel('Components')
axes[0].set_ylabel('Samples')

# Plot S
axes[1].imshow(S, aspect='auto', cmap='viridis')
axes[1].set_title('Matrix S (Diagonal)')
axes[1].set_xlabel('Components')
axes[1].set_ylabel('Components')

# Plot Vt
axes[2].imshow(Vt, aspect='auto', cmap='viridis')
axes[2].set_title('Matrix V^T')
axes[2].set_xlabel('Features')
axes[2].set_ylabel('Components')

plt.tight_layout()
plt.show()

# Reducir la dimensionalidad: Matrices de rango d más cercanas a X.
# Va cambiando las cantidades de variables medidas que va tomando en cuenta.
for d in (2,6,10, data_matrix_X.shape[1]):
    print("Matriz reducida. d = ", d, ".")
    U_d = U[:,:d]
    S_d = S[:d,:d]
    Vt_d = Vt[:d,:]
    
    # Plot U, S, Vt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot U
    axes[0].imshow(U_d, aspect='auto', cmap='viridis')
    axes[0].set_title('Matrix U (First d columns)')
    axes[0].set_xlabel('Components')
    axes[0].set_ylabel('Samples')

    # Plot S
    axes[1].imshow(S_d, aspect='auto', cmap='viridis')
    axes[1].set_title('Matrix S (Diagonal)')
    axes[1].set_xlabel('Components')
    axes[1].set_ylabel('Components')

    # Plot Vt
    axes[2].imshow(Vt_d, aspect='auto', cmap='viridis')
    axes[2].set_title('Matrix V^T (First d rows)')
    axes[2].set_xlabel('Features')
    axes[2].set_ylabel('Components')

    plt.tight_layout()
    plt.show()
    
    X_aprox_d= U_d@S_d@Vt_d
    print("el shape de X_aprox_d es: ", X_aprox_d.shape)
    
    # Crear la mariz del nuevo espacio reducido z:
    A_z = Vt_d.T@Vt_d # . x
    
    # Centrar la data (le resto la media a cada columna):
    X_aprox_d = preprocessing.scale(X_aprox_d)

    # Modelo de PCA a utilizar:
    if (d > cant_variables):
        pca = PCA(n_components= cant_variables) #Te tira un error sino ***
    else:
        pca = PCA(n_components= d)
    X_pca_d = pca.fit_transform(X_aprox_d)
    print("el shape de X_pca_d es: ", X_pca_d.shape)
    
    # Scree plot: Muestra la importancia de cada componente que estamos tomando en cuenta (primeros d) 
    # en la modificación que hace A (tipo lo que veíamos que los componentes modificaban geométricamente).
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()
    
    # Plot de los puntos de PCA: Estructura del de datos multidimensionales (cada punto es una muestra).
    pca_df = pd.DataFrame(X_pca_d, columns=labels)
    
    plt.scatter(pca_df.PC1, pca_df.PC2) 
    #En este caso tomamos en cuenta solo los dos primeros componentes.
    #El porcentaje dice cuánta varianza de los datos originales es captada por cada componente principal.
    #Uno con porcentake más alto de varianza es más importante para entender la estructura de datos.
    # - Por eso cuando usamos los 2000, el PC1 y PC2 no tienen porcentajes tan altos, porque como hay 800 mil otros
    #   componentes más, la influencia sobre esa forma baja no?
    plt.title('PCA Graph')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    #Los clusters (agrupamientos) hablan de similitudes entre ese grupo de muestras.
    #Si los puntos están re dispersos significa que hay mucha variabilidad entre las muestras.
    
  
    plt.show()
    
    euclidean_dist_matrix = euclidean_distances(X_pca_d)

    # Aplicar la transformación de afinidad gaussiana
    sigma = 1.0  # Parámetro de escala
    similarity_matrix = gaussian_similarity(euclidean_dist_matrix, sigma=sigma)
    graficar_matriz_similitud(similarity_matrix, d)



    '''
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
    '''
    


    


