import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Sacar las imágenes del zip-
def read_images_from_zip(zip_filepath):
    images = []
    image_shape = None

    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            # Cuestiones de mac, filtrar directorios, etc.
            if file_name.startswith('__MACOSX/') or file_name.endswith('/'):
                continue

            # Chequear si es una imágen.
            try:
                with zip_ref.open(file_name) as img_file:
                    img = plt.imread(img_file)
                    images.append(img)
                    if image_shape is None:
                        image_shape = img.shape
            except Exception as e:
                print(f"Skipping non-image file: {file_name} - {e}")

    return images, image_shape

# Cargo imágenes del primer zip
images1, image_shape1 = read_images_from_zip('datasets_imgs.zip')

def flatten_images(images):
    flattened_images = [img.flatten() for img in images]
    return np.array(flattened_images)

# SVD
def apply_svd(images, n_components):
    svd = TruncatedSVD(n_components=n_components)
    reduced_data = svd.fit_transform(images)
    return svd, reduced_data

# Reconstrucción de imágenes y visualización
def reconstruct_images(svd, reduced_data, original_shape):
    reconstructed = svd.inverse_transform(reduced_data)
    reconstructed_images = reconstructed.reshape((-1, *original_shape))
    return reconstructed_images

# Visualizar imágenes originales y reconstruidas
def visualize_reconstructed_images(original_images, reconstructed_images, n_images=5):
    plt.figure(figsize=(10, 5))
    for i in range(n_images):
        plt.subplot(2, n_images, i + 1)
        plt.imshow(original_images[i].reshape(image_shape1), cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(reconstructed_images[i], cmap='gray')
        plt.axis('off')
    plt.show()

# Similaridad entre imágenes
def compute_similarity(reduced_data):
    similarity_matrix = cosine_similarity(reduced_data)
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()
    return similarity_matrix

# d óptimo
def find_optimal_d(images, max_d, error_threshold=0.1):
    errors = []
    for d in range(1, max_d+1):
        svd, reduced_data = apply_svd(images, n_components=d)
        reconstructed_images = reconstruct_images(svd, reduced_data, image_shape1)
        error = np.linalg.norm(images - reconstructed_images.reshape(images.shape), 'fro') / np.linalg.norm(images, 'fro')
        errors.append(error)
        if error <= error_threshold:
            return d, errors
    return max_d, errors

# CCargo imágenes del segundo zip
images2, image_shape2 = read_images_from_zip('datasets_imgs_02.zip')

# Aplano las imágenes del primer zip (están en 3 dimensiones por los colores RGB)
images1 = flatten_images(images1)


# 1: SVD de images1
svd1, reduced_data1 = apply_svd(images1, n_components=50)

# 2: Visualizo las imágenes reconstruidas
reconstructed_images1 = reconstruct_images(svd1, reduced_data1, image_shape1)
visualize_reconstructed_images(images1, reconstructed_images1)

# 3: Mido la similaridad entre pares de imágenes
similarity_matrix1 = compute_similarity(reduced_data1)

# Aplano las imágenes del segundo zip (están en 3 dimensiones por los colores RGB)
images2 = flatten_images(images2)

# 4: d óptimo para el segundo conjunto de datos
optimal_d, errors = find_optimal_d(images2, max_d=100)
print(f'Optimal d: {optimal_d}')

# Gráfico error(d)
plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel('Number of Dimensions (d)')
plt.ylabel('Reconstruction Error')
plt.show()

# Aplico la misma compresión a las imágenes del segundo zip.
svd2, reduced_data2 = apply_svd(images2, n_components=optimal_d)
reconstructed_images2 = reconstruct_images(svd2, reduced_data2, image_shape2)
reconstructed_data1_in_svd2 = svd2.transform(images1)
reconstructed_images1_in_svd2 = svd2.inverse_transform(reconstructed_data1_in_svd2).reshape(images1.shape)

# Mido el error de reconstrucción para las imágenes del segundo zip usando la compresión hecha a las del primer zip
reconstruction_error = np.linalg.norm(images1 - reconstructed_images1_in_svd2, 'fro') / np.linalg.norm(images1, 'fro')
print(f'Reconstruction error for dataset_imagenes1 using SVD from dataset_imagenes2: {reconstruction_error}')
