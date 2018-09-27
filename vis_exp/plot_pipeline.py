import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from utils import imshow_util_uint8
from datasets.cwr_dataset import CWR_Dataset
from datasets.dataset import Dataset



def plot_interactive(f_2d,indexs,dataset_obj : Dataset,labels):

    n=f_2d.shape[0]
    d_point_index = {i : indexs[i] for i in range(f_2d.shape[0])}
    img=imshow_util_uint8(dataset_obj.get_train_image_at(indexs[0])[0][0],dataset_obj)

    # create figure and plot scatter
    fig,axis = plt.subplots(1, 2)
    line = axis[0].scatter(f_2d[:,0],f_2d[:,1],c=labels.reshape(n),s=5)
    temp= axis[1].imshow(img)


    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            ind = line.contains(event)[1]["ind"][0]
            r_ind = d_point_index[ind]
            print("Getting {0} index".format(r_ind))
            img = imshow_util_uint8(dataset_obj.get_train_image_at(r_ind)[0][0], dataset_obj)
            temp.set_data(img)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()

def pca_pipeline(features,indexs,labels,dataset,out_path=None,interactive=False):
    assert(len(features.shape) == 2),'Features must be matrix'

    pca=PCA(n_components=2)
    f_2d = pca.fit_transform(features)
    n= f_2d.shape[0]
    print("Explained var_r : {0}".format(pca.explained_variance_ratio_))

    if interactive:
        plot_interactive(f_2d, indexs, dataset, labels)

    if out_path:
        plt.scatter(f_2d[:, 0], f_2d[:, 1], c=labels.reshape(n), s=5)
        plt.savefig(out_path)

    pass





if __name__ == '__main__':
    import random
    x=np.random.rand(1000,100)
    y = np.random.randint(0,3,1000,np.int)
    dataset = CWR_Dataset(1,1,data_folder='./temp/CW96Scalograms')
    indexs = dataset.get_index_list()
    chosen = random.sample(indexs,1000)


    pca_pipeline(x,chosen,y,dataset,'./temp/pca.png',interactive=True)
    pass