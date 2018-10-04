import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import os
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import numpy as np
from scipy.misc import imresize
import pickle
# cargar cams
from sklearn.decomposition import PCA


#CWR_Clasifier_04_Oct_2018__17_38
from datasets.cwr_dataset import CWR_Dataset
from vis_exp.plot_pipeline import plot_interactive

prefijo = "CWR_Clasifier_04_Oct_2018__18_12"

cam_shape = (96,96)
path_pickle_indexs = "vis_results/{0}/indexs.pkl".format(prefijo)
path_pickle_labels = "vis_results/{0}/labels.npy".format(prefijo)
path_pickle_cams = "vis_results/{0}/vis_img.npy".format(prefijo)

with open(path_pickle_indexs,'rb') as f:
    indexs = pickle.load(f)
cams_flatten = np.load(path_pickle_cams)
labels = np.load(path_pickle_labels)
cams_raw = [img.reshape(cam_shape) for img in cams_flatten]

#
# folder_cams_raw = "out_backprops/30_Sep_2018__22_03/raw_cams"
# numpy_f_list = os.listdir(folder_cams_raw)
# cams_raw = [np.load(os.path.join(folder_cams_raw,f_path)) for f_path in numpy_f_list]


# prepare filter bank kernels
kernels = []


theta_params = [0,1,2,3]
sigma_params = [1,3]
f,ax = plt.subplots(len(theta_params),len(sigma_params))

for ind_theta,theta in enumerate(theta_params):
    theta = theta / 4. * np.pi
    for ind_sigma,sigma in enumerate(sigma_params):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
            ax[ind_theta,ind_sigma].imshow(kernel)
plt.show()



# por cada imagen calcular activacion
dataset = []
r_labels = []
indexs_names = []

for ind,cam_img in enumerate(cams_raw):

    if ind != 0 and ind % 100 == 0:
        print(ind)

    current_vector = []

    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(cam_img, kernel, mode='wrap')

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
    r_labels.append(np.argmax(labels[ind]))
    indexs_names.append(indexs[ind].decode('utf8'))

dataset = np.vstack(dataset)
# Realizar PCA2D del dataset.
from sklearn.manifold import TSNE

pca = PCA(n_components=2)
dim_rediction_alg = TSNE(n_components=2)
data_x = dim_rediction_alg.fit_transform(dataset)

# plt.scatter(data_x[:,0],data_x[:,1],c=r_indexs)
# plt.show()

plot_interactive(data_x,indexs_names,CWR_Dataset(4,60,data_folder="./temp/CW96Scalograms"),np.array(r_labels))