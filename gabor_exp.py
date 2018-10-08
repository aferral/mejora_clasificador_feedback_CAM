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
from sklearn.manifold import TSNE

#CWR_Clasifier_04_Oct_2018__17_38
from datasets.cwr_dataset import CWR_Dataset
from datasets.dataset import placeholder_dataset, Dataset
from vis_exp.plot_pipeline import plot_interactive

def gabor_features_raw_data(kernels,dataset_obj : Dataset,img_shape):
    limit = 1400

    indexs=[]
    labels=[]
    imgs_raw=[]
    cont=0
    for indx in dataset_obj.get_index_list():


        img,label = dataset_obj.get_train_image_at(indx)
        # if not(label in ['1','3']):
        #     continue
        indexs.append(indx)
        img = img / 255
        labels.append(int(label))
        imgs_raw.append(img.reshape(img_shape) )
        cont+=1
        if cont > (limit):
            break



    # por cada imagen calcular activacion
    dataset = []
    r_labels = []
    indexs_names = []

    for ind, cam_img in enumerate(imgs_raw):
        if ind > len(labels):
            break

        if ind != 0 and ind % 100 == 0:
            print(ind)

        current_vector = []

        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(cam_img, kernel, mode='wrap')

            # reducir a 8x8
            res = imresize(filtered, (8, 8))

            # convertir a vector
            filter_vector = res.flatten()

            current_vector.append(filter_vector)


        # concadenar por todos los filtros
        current_vector = np.array(current_vector)
        current_vector = current_vector.flatten()

        # agrega como un vector al dataset
        dataset.append(current_vector)
        r_labels.append((labels[ind]))
        ind_to_add = indexs[ind].decode('utf8') if type(indexs[ind]) == bytes else indexs[ind]
        indexs_names.append(ind_to_add)

    dataset = np.vstack(dataset)
    return dataset,indexs_names,r_labels,imgs_raw

def gabor_features_cam_data(kernels,folder_path,cam_shape):
    path_pickle_indexs = "vis_results/{0}/indexs.pkl".format(folder_path)
    path_pickle_labels = "vis_results/{0}/labels.npy".format(folder_path)
    path_pickle_cams = "vis_results/{0}/vis_img.npy".format(folder_path)

    with open(path_pickle_indexs, 'rb') as f:
        indexs = pickle.load(f)
    cams_flatten = np.load(path_pickle_cams)
    labels = np.load(path_pickle_labels)
    cams_raw = [img.reshape(cam_shape) for img in cams_flatten]


    # por cada imagen calcular activacion
    dataset = []
    r_labels = []
    indexs_names = []
    out_cams = []

    for ind, cam_img in enumerate(cams_raw):

        if ind != 0 and ind % 100 == 0:
            print(ind)

        label=np.argmax(labels[ind])
        # print("l: {0}".format(label))
        if not(label in [1,3]):
            continue


        current_vector = []

        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(cam_img, kernel, mode='wrap')
            # reducir a 8x8
            res = imresize(filtered, (8, 8))
            # convertir a vector
            filter_vector = res.flatten()
            current_vector.append(filter_vector)


        # concadenar por todos los filtros
        current_vector = np.array(current_vector)
        current_vector = current_vector.flatten()

        # agrega como un vector al dataset
        dataset.append(current_vector)
        r_labels.append(label)
        indexs_names.append(indexs[ind].decode('utf8'))
        out_cams.append(cam_img)

    dataset = np.vstack(dataset)
    return dataset,indexs_names,r_labels,out_cams

def gabor_features_data(kernels,folder_path,cam_shape):
    dataset, indexs_names, r_labels, cams_raw = gabor_features_cam_data(kernels,folder_path, cam_shape)
    return dataset,indexs_names,r_labels




if __name__ == '__main__':
    prefijo = "CWR_Clasifier_04_Oct_2018__18_12"
    cam_shape = (96,96)
    dataset_obj = CWR_Dataset(4,60,data_folder="./temp/CW96Scalograms")

    dataset_one_use = placeholder_dataset(dataset_obj)

    load_old = False



    # # prepare filter bank kernels
    kernels = []

    theta_params = [0]
    sigma_params = [1, 3,6]


    # prepare filter bank kernels
    # theta_params = [0]
    # sigma_params = [1,3,6,8]


    f, ax = plt.subplots(len(theta_params), len(sigma_params))
    ax = [[x for x in ax]] if len(theta_params) == 1 else ax

    for ind_theta, theta in enumerate(theta_params):
        theta = theta / 2. * np.pi
        for ind_sigma, sigma in enumerate(sigma_params):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
                ax[ind_theta][ind_sigma].imshow(kernel)
    plt.show()




    if load_old:
        with open("temp.pkl", 'rb') as f:
            temp_d = pickle.load(f)
        dataset, indexs_names, r_labels,vis = temp_d

    else:
        dataset, indexs_names, r_labels,vis = gabor_features_cam_data(kernels,prefijo,cam_shape)
        # dataset, indexs_names, r_labels, vis = gabor_features_raw_data(kernels,dataset_obj,cam_shape)

        temp_d = [dataset, indexs_names, r_labels,vis]
        with open("temp.pkl",'wb') as f:
            pickle.dump(temp_d,f)


    pca = PCA(n_components=2)
    dim_rediction_alg = TSNE(n_components=2)

    vis_uint8 = [(img*255).astype(np.uint8) for img in vis]
    dataset_one_use.prepare_dataset(indexs_names,vis_uint8 , r_labels)
    data_x = dim_rediction_alg.fit_transform(dataset)

    plot_interactive(data_x, indexs_names, dataset_one_use, np.array(r_labels))


