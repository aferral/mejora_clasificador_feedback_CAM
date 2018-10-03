import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import os
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import numpy as np
from scipy.misc import imresize

# cargar cams
from sklearn.decomposition import PCA

folder_cams_raw = "out_backprops/30_Sep_2018__22_03/raw_cams"
numpy_f_list = os.listdir(folder_cams_raw)
cams_raw = [np.load(os.path.join(folder_cams_raw,f_path)) for f_path in numpy_f_list]
print('hi')


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


# por cada imagen calcular activacion
dataset = []
for cam_index_x in cams_raw:
    selected_index = 0
    selected_cam =  cam_index_x[selected_index]

    current_vector = []

    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(selected_cam, kernel, mode='wrap')

        # reducir a 8x8
        res = imresize(filtered,(8,8))

        # convertir a vector
        filter_vector = res.flatten()

        current_vector.append(filter_vector)


        # plt.figure()
        # plt.imshow(kernel)
        # plt.figure()
        # plt.imshow(res)
        # plt.show()

    # concadenar por todos los filtros
    current_vector = np.array(current_vector)
    current_vector = current_vector.flatten()

    # agrega como un vector al dataset
    dataset.append(current_vector)

dataset = np.vstack(dataset)
# Realizar PCA2D del dataset.

pca = PCA(n_components=2)
data_x = pca.fit_transform(dataset)

plt.scatter(data_x[:,0],data_x[:,1])
plt.show()