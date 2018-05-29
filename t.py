from contextlib import ExitStack

import tensorflow as tf
import numpy as np
import os

# From http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
from classification_models.classification_model import imshow_util
from classification_models.vgg16_edited import vgg_16_CAM
from datasets.cifar10_data import Cifar10_Dataset
from datasets.dataset import Dataset, Digits_Dataset
from datasets.imagenet_data import Imagenet_Dataset
from utils import show_graph, now_string, timeit


"""
# Install

https://unix.stackexchange.com/questions/332641/how-to-install-python-3-6

# Download data VOC 2017

train val
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

test
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

sudo apt-get install libbz2-dev libncurses5-dev libgdbm-dev liblzma-dev sqlite3 libsqlite3-dev openssl libssl-dev tcl8.6-dev tk8.6-dev libreadline-dev zlib1g-dev      


sudo apt-get install python-setuptools python-dev ncurses-dev


pip install readline

python -m ipykernel install --user --name GSDA_env --display-name "GSDA_env"


# visualizar, seleccionar, generar, ajustar


class model(entrenar / load)
load-model
feed-forward
visualize
train
re_train


(jupyter)
- Dado dataset object, classifier entregar lista de elementos mal clasificados (val set)

- Dado imagen mostrar mapas de clase para cada clase con su porcentaje prob.
- Select interactivo de partes sde imagen
- Guarda lista de mascaras binaria,imagenes a archivo 


find_errors select(img) -> bin_mask
- Genera lista de imagenes evaluando dataset
- Busca imagenes mal clasificadas
- Muestra mapas de visualizacion
- Permite hacer el select



generator(img_list) -> new_img_list
- Con lista de imagenes y mascaras comienza a genear imagenes nuevas
- Entrega un dataset augmentado

re_train
- Dado un objeto dataset.


Notas a mejorar
- Uso de models slim???
- Elegir requirement con o sin gpu?
- Informacion importante tener muchos cudas
sudo sh cuda-9.1.run --silent --toolkit --toolkitpath=/usr/local/cuda-9.0

export PATH=/home/aferral/cuda-9.0/bin:$PATH ;
export LD_LIBRARY_PATH=/home/aferral/cuda-9.0/lib64:$LD_LIBRARY_PATH


hacer mas facil seteo de variables y entornos en pychar o lo que sea 

"""
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    train=False

    # todo improve dataset get
    # todo add submodules for tf_models
    # start to work on mnist or VOC

    # dataset = Digits_Dataset(epochs=20,batch_size=30)
    dataset = Cifar10_Dataset(20,40)
    # dataset = Imagenet_Dataset(20,30)

    with vgg_16_CAM(dataset, debug=False) as model:

        if train:
            model.train()
        else:
            # model.load('./model/check.meta','model/cifar10_classifier/23_May_2018__10_54')
            #model.load('./model/check.meta', 'model/digit_classifier/24_May_2018__15_48')
            model.load('./model/check.meta','./model/vgg16_classifier/29_May_2018__01_41')
            # model.eval()


            test_image = dataset.get_train_image_at(0)[0]
            test_image_plot = imshow_util( test_image.reshape(dataset.vis_shape()),dataset.get_data_range())

            image_processed, prediction, cmaps = model.visualize(test_image)

            image_processed_plot = imshow_util( image_processed.reshape(dataset.vis_shape()),dataset.get_data_range())

            p_class = np.argmax(prediction)
            print("Predicted {0} with score {1}".format(p_class,np.max(prediction)))
            print(cmaps.shape)
            print("CMAP: ")

            import matplotlib.pyplot as plt
            from skimage.transform import resize


            plt.figure()
            plt.imshow(image_processed_plot,cmap='gray')

            plt.figure()
            plt.imshow(test_image_plot,cmap='gray')


            plt.figure()
            plt.imshow(cmaps[0],cmap='jet',interpolation='none')

            out_shape = list(test_image_plot.shape)
            if len(test_image_plot.shape) == 3:
                out_shape = out_shape[0:2]
            print(out_shape)
            resized_map = resize(cmaps[0],out_shape)
            plt.figure()
            plt.imshow(resized_map,cmap='jet')

            fig, ax = plt.subplots()
            ax.imshow(resized_map, cmap='jet',alpha=0.7)
            ax.imshow(image_processed_plot,alpha=0.3,cmap='gray')
            plt.show()
