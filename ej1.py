import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD

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

def gaussian_similarity(dist_matrix, sigma=1.0):
    # Convertir la matriz de distancia en matriz de similitud usando afinidad gaussiana
    return np.exp(-dist_matrix**2 / (2 * sigma**2))

def desempaquetado_datos():
    # Desempaquetado de los datos en el archivo csv, guardados en una matriz
    data_matrix_X = np.loadtxt('dataset03.csv', delimiter=",", skiprows=1, usecols=range(1, np.genfromtxt('dataset03.csv', delimiter=",", max_rows=1).size))
    cant_filas = data_matrix_X.shape[0]
    cant_variables = data_matrix_X.shape[1]
    y = np.loadtxt('y3.txt')
    print("cantidad de muestras: ", cant_filas,". cantidad de features: ", cant_variables)
    
    return data_matrix_X, y

def SVD(X):
    # Descomposición SVD:
    U, S, Vt = np.linalg.svd(X, full_matrices=False) 
    S = np.diag(S) #svd me devuelve un vector de valores singulares, lo convierto a matriz.
    
    return U, S, Vt

def spearman_2(data, d):
    # Calcular las correlaciones de Spearman
    correlations = np.zeros((data.shape[1], 5))  # Matriz para almacenar las correlaciones

    for i in range(5):  # Para cada componente principal
        for j in range(data.shape[1]):  # Para cada dimensión original
            correlations[j, i], _ = spearmanr(data[:, j], components[:, i])

    # Crear un DataFrame para las correlaciones
    correlations_df = pd.DataFrame(correlations, columns=[f'PC{i+1}' for i in range(5)], index=[f'Feature_{i+1}' for i in range(data.shape[1])])

    # Identificar las 5 dimensiones con mayor correlación para cada componente principal
    top_features = {}
    for col in correlations_df.columns:
        top_features[col] = correlations_df[col].abs().nlargest(5).index.tolist()

    # Crear un DataFrame para las correlaciones de las top features
    top_correlations = correlations_df.loc[np.unique(sum(top_features.values(), []))]

    # Graficar las correlaciones usando un heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Top 5 Correlaciones de Spearman entre Dimensiones Originales y Componentes Principales')
    plt.xlabel('Componentes Principales')
    plt.ylabel('Dimensiones Originales')
    plt.show()

def coeficiente_spearman_calc(S, X, X_d):
    
    # COEFICIENTE SPEARMAN: Qué tan bien se preserva la información de la matriz original.
    # Valores más bajos implica que las matrices se parecen menos, valores altos que se parecen más.
    
    energía_acumulada = np.cumsum(S**2) / np.sum(S**2)
    umbral = 0.95  
    # Para retener el 95% de la variabilidad (energia/info total contenida 
    # en los datos originales al reducir la dimensionalidad con SVD).
    
    # Variabilidad de los datos: Captada por los valores singulares
    k = np.where(energía_acumulada >= umbral)[0][0] + 1
    correlaciones = [spearmanr(X[i], X_d[i]).correlation for i in range(X.shape[0])]
    correlación_promedio = np.mean(correlaciones)
    return correlación_promedio

def PCA_mine(X_d_scaled, d, cant_features):
    
    # Modelo de PCA a utilizar:
    if (d > cant_features):
        pca = PCA(n_components= cant_features) # Atajo el caso borde
    else:
        pca = PCA(n_components= d)
        
    X_d_pca = pca.fit_transform(X_d_scaled)
    
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    
    # SCREE PLOT: Muestra la importancia de cada componente que estamos tomando en cuenta (primeros d) 
    # en la modificación que hace A (tipo lo que veíamos que los componentes modificaban geométricamente).
    # Representa los valores de la varianza explicada por cada componente principal o factor
    
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()
    
    
    # PLOT PCA, PRIMEROS DOS COMPONENTES: Estructura del de datos multidimensionales (cada punto es una muestra).
    pca_df = pd.DataFrame(X_d_pca, columns=labels)
    
    # En este caso tomamos en cuenta solo los dos primeros componentes.
    # El porcentaje dice cuánta varianza de los datos originales es captada por cada componente principal.
    # Uno con porcentaje más alto de varianza es más importante para entender la estructura de datos.
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('PCA Graph')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    # Los porcentajes representan la proporción de la varianza total que es explicada por cada componente principal
    # Los clusters (agrupamientos) hablan de similitudes entre ese grupo de muestras.
    # Si los puntos están re dispersos significa que hay mucha variabilidad entre las muestras.
    
    plt.show()
    
    return X_d_pca

def matriz_carga(d, X, cant_features):
    # Matriz de carga: indica cuánto contribuye cada variable original a cada uno de los componentes principales. Es una forma de entender cómo se relacionan las dimensiones originales del dataset con las nuevas dimensiones obtenidas (componentes principales).
    svd = TruncatedSVD(n_components=d)
    X_svd = svd.fit_transform(X)
    V = svd.components_  # Componentes principales
    Sigma = svd.singular_values_  # Valores singulares

    Lambda = np.dot(V.T, np.diag(Sigma))
    # Cada fila corresponde a una variable original y cada columna a un componente principal. Los valores absolutos en esta matriz indican la importancia de cada variable original para cada componente principal.

    # Identificar las dimensiones originales más importantes: Sumo las cargas absolutas para 
    # cada variable original a través de todos los componentes principales seleccionados y ordeno 
    # estos valores de mayor a menor.

    # Sumar las cargas absolutas para cada variable original
    most_important_features = np.sum(np.abs(Lambda), axis=1)

    features = np.array([f'Dimension {i+1}' for i in range(cant_features)])

    # Ordenar las características por importancia
    sorted_indices = np.argsort(most_important_features)[::-1]
    sorted_features = features[sorted_indices]
    sorted_importance = most_important_features[sorted_indices]

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_features, sorted_importance)
    plt.title('Importancia de cada dimensión original respecto a las componentes principales')
    plt.xlabel('Dimensiones originales')
    plt.ylabel('Suma de las cargas absolutas')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Las dimensiones originales con los mayores valores absolutos en la matriz de cargas son las 
    # más importantes porque tienen la mayor contribución a la variabilidad capturada por los 
    # componentes principales seleccionados. Estas dimensiones son las que más influyen en la 
    # estructura subyacente del conjunto de datos tal como está representado por los componentes 
    # principales.

def Graf_U_S_Vt(U_d, S_d, Vt_d):
    
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
       
def cuadrados_minimos(U, S, Vt, y):
    
    matrix = U@S@Vt
    pseudoinv = np.linalg.pinv(matrix)
    x = np.dot(pseudoinv, y)
    
    y_aprox = np.dot(matrix, x)
    
    error = np.linalg.norm(y_aprox - y)
    print("error: ", error)
    
    indices_y = np.arange(len(y))
    plt.scatter(indices_y, y, color='b', label='y real')
    plt.scatter(indices_y, y_aprox, color='red', label='y aprox')
    plt.title("y (aproximación) vs y")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()
        
    return error
       
def plot_last_and_first_vs(U,S,Vt, num):
    original_X = U@S@Vt
    first_num_vs_X = U[:, :num]@S[:num, :num]@Vt[:num, :]
    last_num_vs_X = U[:, -num:]@S[-num:, -num:]@Vt[-num:, :]
    
    # Plot original, first y last.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original
    axes[0].imshow(original_X, aspect='auto', cmap='viridis')
    axes[0].set_title('Matriz original')
    axes[0].set_xlabel('Components')
    axes[0].set_ylabel('Samples')

    # Plot first
    axes[1].imshow(first_num_vs_X, aspect='auto', cmap='viridis')
    axes[1].set_title(f'Matriz: Primeros {num} valores singulares')
    axes[1].set_xlabel('Components')
    axes[1].set_ylabel('Samples')
    
    # Plot last
    axes[2].imshow(last_num_vs_X, aspect='auto', cmap='viridis')
    axes[2].set_title(f'Matriz: Últimos {num} valores singulares')
    axes[2].set_xlabel('Components')
    axes[2].set_ylabel('Samples')

    plt.tight_layout()
    plt.show()
    
def main():
    
    X, y = desempaquetado_datos()
    cant_filas = X.shape[0]
    cant_features = X.shape[1]
    
    # Centrado de datos:
    y = y - np.mean(y)
    col_mean = np.mean(X, axis = 0)
    X = X - col_mean
    
    
    # Creo estructuras para guardar datos para graficar mas adelante.
    dimension_arr = []
    correl_spearman_different_dimensions = []
    err_aprox_y_arr = []
    
    
    # SVD:
    U, S, Vt = SVD(X)

    # Recorte de dimensiones:
    for d in (2,6,10, X.shape[1]):
        
        #plot_last_and_first_vs(U, S, Vt, d)
        
        print("Matriz reducida. d = ", d, ".")
        U_d = U[:,:d]
        S_d = S[:d,:d]
        Vt_d = Vt[:d,:]
        
        # Reconstrucción de X con la versión reducida a d dimensiones:
        X_d = U_d@S_d@Vt_d
        
        # SPEARMAN: 
        correlación_promedio = coeficiente_spearman_calc(S, X, X_d)
        dimension_arr.append(d)
        correl_spearman_different_dimensions.append(correlación_promedio)

        # PCA:
        X_d_pca = PCA_mine(X_d, d, cant_features)
        
        # MATRIZ DE SIMILITUD:
        euclidean_dist_matrix = euclidean_distances(X_d_pca)
        similarity_matrix = gaussian_similarity(euclidean_dist_matrix, sigma= 1.0)
        graficar_matriz_similitud(similarity_matrix, d)
        
        # MATRIZ DE CARGA:
        #matriz_carga(d)
        
        # CUADRADOS MÍNIMOS, PROYECCIÓN:
        error_curr_dim = cuadrados_minimos(U_d, S_d, Vt_d, y)
        err_aprox_y_arr.append(error_curr_dim)
        
        
    # Gráfico del coeficiente de correlación de spearman para diferentes d:
    plt.plot(dimension_arr, correl_spearman_different_dimensions)
    plt.scatter( dimension_arr, correl_spearman_different_dimensions)
    plt.title("Coeficiente de correlación de spearman - Número de dimensiones")
    plt.xlabel("Número de dimensiones")
    plt.ylabel("Coeficiente de correlación de spearman")
    plt.grid(True)
    plt.show()
    
    # Gráfico del coeficiente de correlación de spearman para diferentes d:
    plt.plot(dimension_arr, err_aprox_y_arr)
    plt.scatter( dimension_arr, err_aprox_y_arr)
    plt.title(" Error de aproximación - Número de dimensiones")
    plt.xlabel("Número de dimensiones")
    plt.ylabel("|| Ax - y||2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()



