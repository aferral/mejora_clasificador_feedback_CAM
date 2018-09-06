from contextlib import ExitStack

import tensorflow as tf
import numpy as np
import os

# From http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
from classification_models.classification_model import imshow_util, \
    CWR_classifier
from classification_models.vgg16_edited import vgg_16_CAM
from classification_models.vgg_16_batch_norm import vgg_16_batchnorm
from datasets.cifar10_data import Cifar10_Dataset
from datasets.cwr_dataset import CWR_Dataset
from datasets.dataset import Dataset, Digits_Dataset
from datasets.imagenet_data import Imagenet_Dataset
from select_tool.config_data import model_obj_dict, dataset_obj_dict
from utils import show_graph, now_string, timeit
import json

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

isntale line profiler

revisar el api de dataset nuevamente

Cuidaod con que repitan validation set

IMPORTANTE PENSAR ACERCA DE CAMBIAR DATOS DE TF RECORDS Y SOLO USAR TFRECORD para entrenar????

todo mecanizar download cwr

TODO revisar como llevar a tf recrods CWR bien (multiplica size 10 veces)

"""
import tensorflow as tf
import matplotlib.pyplot as plt

def train_for_epochs(dataset,model_class,model_params,load_path,train_file_path):

    with model_class(dataset,**model_params) as model:

        if load_path:
            model.load(load_path)

        model.train(train_file_used=train_file_path)



def do_train_config(config_path):
    with open(config_path,'r') as f:
        data=json.load(f)

    t_mode = data["train_mode"]
    t_params = data['train_params']
    model_key = data['model_key']
    model_params = data['model_params']
    model_load_path = data['model_load_path']
    dataset_key = data['dataset_key']
    dataset_params = data['dataset_params']

    batch_size = t_params['b_size'] if 'b_size' in t_params else 20
    epochs = t_params['epochs'] if 'epochs' in t_params else 1

    if t_mode == 'epochs':
        model_class = model_obj_dict[model_key]
        dataset_class = dataset_obj_dict[dataset_key]
        dataset_obj = dataset_class(epochs,batch_size,**dataset_params)
        train_for_epochs(dataset_obj,model_class,model_params,model_load_path,config_path)



if __name__ == '__main__':
    import argparse

    # generate gen_file (carpeta_con_imagenes, select_file_usado, modelo_usado )
    # todo crear algoritmo de train especializado a gene file
    # todo testear el sistema con caso CWR o cifar10

    parser = argparse.ArgumentParser(description='Execute train config ')
    parser.add_argument('train_config_json', help='The config_json to train')

    args = parser.parse_args()
    do_train_config(args.train_config_json)


#     # dataset = Digits_Dataset(epochs=20,batch_size=30)
#     # dataset = Cifar10_Dataset(20,40)
#     dataset = Imagenet_Dataset(20,30)
#
#
#             test_image,labels = dataset.get_train_image_at(0)[0]
#             test_image_plot = imshow_util( test_image.reshape(dataset.vis_shape()),dataset.get_data_range())
#
#             image_processed, prediction, cmaps = model.visualize(test_image)
#
#             image_processed_plot = imshow_util( image_processed.reshape(dataset.vis_shape()),dataset.get_data_range())
#
#             p_class = np.argmax(prediction)
#             print("Predicted {0} with score {1}".format(p_class,np.max(prediction)))
#             print(cmaps.shape)
#             print("CMAP: ")
#
#             import matplotlib.pyplot as plt
#             from skimage.transform import resize
#
#
#             plt.figure()
#             plt.imshow(image_processed_plot,cmap='gray')
#
#             plt.figure()
#             plt.imshow(test_image_plot,cmap='gray')
#
#
#             plt.figure()
#             plt.imshow(cmaps[0],cmap='jet',interpolation='none')
#
#             out_shape = list(test_image_plot.shape)
#             if len(test_image_plot.shape) == 3:
#                 out_shape = out_shape[0:2]
#             print(out_shape)
#             resized_map = resize(cmaps[0],out_shape)
#             plt.figure()
#             plt.imshow(resized_map,cmap='jet')
#
#             fig, ax = plt.subplots()
#             ax.imshow(resized_map, cmap='jet',alpha=0.7)
#             ax.imshow(image_processed_plot,alpha=0.3,cmap='gray')
#             plt.show()
# with vgg_16_batchnorm(dataset, debug=False, name='Imagenet_subset_vgg16_CAM') as model:
#     if train:
#         model.train()
#     else:
#         # model.load('./model/check.meta','model/cifar10_classifier/23_May_2018__10_54')
#         # model.load('./model/check.meta', 'model/digit_classifier/24_May_2018__15_48')
#         model.load('./model/vgg16_classifier/29_May_2018__01_41')
        # model.eval()