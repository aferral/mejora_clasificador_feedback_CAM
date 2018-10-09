from contextlib import ExitStack
import random
import cv2
import tensorflow as tf
import numpy as np
import os

# From http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
from classification_models.classification_model import CWR_classifier
from classification_models.vgg16_edited import vgg_16_CAM
from classification_models.vgg_16_batch_norm import vgg_16_batchnorm
from datasets.cifar10_data import Cifar10_Dataset
from datasets.cwr_dataset import CWR_Dataset
from datasets.dataset import Dataset, Digits_Dataset, placeholder_dataset
from image_generator.factory import get_generator_from_key
from image_generator.yu2018 import yu2018generative
from select_tool.config_data import model_obj_dict, dataset_obj_dict
from select_tool.img_selector import call_one_use_select

from utils import show_graph, now_string, timeit, load_mask, imshow_util, get_img_cam_index, get_img_RAW_cam_index
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


# todo que hacer con labels
def create_lists(dataset, gen_map):
    images = []
    labels = []
    index_list = []
    for img_index in gen_map:  # iterate the generated images adding to dataset
        # Place the original image
        original_img, original_label = dataset.get_train_image_at(img_index)
        images.append(original_img[0])
        labels.append(int(original_label))
        index_list.append(img_index)

        image_path_list = gen_map[img_index]
        for ind_gen, img_path in enumerate(sorted(image_path_list)):
            img = cv2.imread(img_path)
            images.append(img)
            labels.append(int(original_label))
            index_list.append("{0}_gen{1}".format(img_index, ind_gen))

    n_normal = 2
    all_indexs = dataset.get_index_list()
    for i in range(n_normal):
        ind_c = random.choice(all_indexs)
        original_img, original_label = dataset.get_train_image_at(ind_c)
        images.append(original_img[0])
        labels.append(int(original_label))
        index_list.append(ind_c)
    return images, labels, index_list



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


    if t_mode == 'epochs':

        model_key = data['model_key']
        model_params = data['model_params']
        model_load_path = data['model_load_path']
        dataset_key = data['dataset_key']
        dataset_params = data['dataset_params']

        batch_size = t_params['b_size'] if 'b_size' in t_params else 20
        epochs = t_params['epochs'] if 'epochs' in t_params else 1
        epochs = 1; # todo test code

        model_class = model_obj_dict[model_key]
        dataset_class = dataset_obj_dict[dataset_key]
        base_dataset = dataset_class(epochs,batch_size,**dataset_params)
        train_for_epochs(base_dataset,model_class,model_params,model_load_path,config_path)

    elif t_mode == "gen_train":
        gen_file = t_params['gen_file']

        with open(gen_file) as f:
            data_gen=json.load(f)


        used_select = data_gen['used_select']
        with open(used_select) as f:
            data_select=json.load(f)
        train_result_path = data_select['train_result_path']
        with open(train_result_path,'r') as f:
            data_t_r = json.load(f)
            path_train_file = data_t_r["train_file_used"]

            with open(path_train_file,'r') as f2:
                data_train_file = json.load(f2)
            m_k = data_train_file['model_key']
            m_p = data_train_file['model_params']
            d_k = data_train_file['dataset_key']
            d_p = data_train_file['dataset_params']

        if data['model_load_path'] is not None:
            model_load_path = data['model_load_path']
            print("Using model load path from TRAIN_FILE {0}".format(model_load_path))
        else:
            model_load_path = data_t_r['model_load_path']
            print("Using model load path from SELECT_FILE {0}".format(model_load_path))


        batch_size = t_params['b_size'] if 'b_size' in t_params else 20
        epochs = t_params['epochs'] if 'epochs' in t_params else 1

        model_class = model_obj_dict[m_k]
        dataset_class = dataset_obj_dict[d_k]
        base_dataset = dataset_class(epochs,batch_size,**d_p) # type: Dataset


        if ('just_eval' in t_params) and (t_params['just_eval']):
            print("Doing eval for validation set")
            with model_class(base_dataset, **m_p) as model:
                if model_load_path:
                    model.load(model_load_path)

                model.eval(mode='test')
            return


        # images, labels, index_list = create_lists(base_dataset, data_gen['index_map'])
        # Create dummy dataset add all gen_images and random images
        dataset_one_use = placeholder_dataset(base_dataset)


        def eval_current_model(name,classifier, dataset, ind_image,ind_backprop,out_f,current_log,set_cam_index,eval=False):

            os.makedirs(out_f,exist_ok=True)
            path_out_cams=os.path.join(out_f,'raw_cams')
            os.makedirs(path_out_cams,exist_ok=True)


            if eval:
                out_string = classifier.eval(mode='test', samples=10)
                with open(os.path.join(out_f,'eval.txt'),'a') as f:
                    f.write(current_log)
                    f.write(out_string)


            img, all_cams, scores, r_label,raw_cams = get_img_RAW_cam_index(dataset, classifier, ind_image)
            print(scores)
            np.save(os.path.join(path_out_cams,
                                 '{1}_raw_vis_it_{0}.npy'.format(ind_backprop,
                                                                 name)),
                    raw_cams)



            ind_selected = int(set_cam_index)
            img_colored = cv2.cvtColor(
                cv2.applyColorMap(all_cams[ind_selected], cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB)
            plt.imshow(img_colored)
            plt.savefig(os.path.join(out_f, 'Selected_{0}.png'.format(
                ind_backprop)))


            plt.clf()
            f, axs = plt.subplots(1, len(all_cams))
            for ind in range(len(all_cams)):
                # important cv2 make a BGR transform to RGB for matplotlib

                img_colored = cv2.cvtColor(cv2.applyColorMap(all_cams[ind],cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
                l=r_label
                scor="{0:.2f}".format(scores[ind])

                axs[ind].set_title('R_l {0} Cls {1} -- {2}'.format(l,ind,scor),fontdict={'fontsize': 11})
                axs[ind].imshow(img_colored)

            plt.savefig(os.path.join(out_f,'{1}__it_{0}.png'.format(ind_backprop,name)), bbox_inches='tight', pad_inches=0)

        # accion: seleccionar mascara desde (img,img_cam) -> mask
        def sel_mask(img_cam, img_or):
            mask = call_one_use_select(img_cam, img_or=img_or)
            plt.figure()
            plt.imshow(mask)
            plt.show()
            return mask

        # accion: selecciona indice (indice) --> img,img_cams
        def get_img_cam(index_image, dataset, classifier,ind):
            img, label = dataset.get_train_image_at(index_image)
            img_proc, all_cams, scores, r_label,raw_cams = get_img_RAW_cam_index(dataset, classifier, ind)

            plt.figure()
            plt.title('img_selected')
            plt.imshow(img_proc)

            f, axs = plt.subplots(1, len(all_cams))
            for ind in range(len(all_cams)):
                # Remember that cv2 make a BGR. Transform to RGB for matplotlib
                img_colored = cv2.cvtColor(cv2.applyColorMap(all_cams[ind],cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
                l=r_label
                scor="{0:.2f}".format(scores[ind])
                axs[ind].set_title('R_label: {0} Class {1} -- {2}'.format(l,ind,scor))
                axs[ind].imshow(img_colored)
            # f.tight_layout()
            plt.show()

            return img,label,all_cams,scores,r_label

        with model_class(dataset_one_use, **m_p) as model: # this also save the train_result
            model.load(model_load_path)
            # dataset_one_use.prepare_dataset(index_list, images, labels)
            # dataset_one_use.show_current()
            # model.train(train_file_used=config_path,save_model=False,eval=True)


            current_ind = None
            current_img=None
            current_label=None
            current_cams=None
            selected_cam=0
            current_mask=None
            gen_images=None
            backprops=0
            st_gen_index = None
            gens=0

            index_list = []
            img_list = []
            label_list = []

            # accion: invocar ref_gen
            gen_model = get_generator_from_key("random_crop",dataset=base_dataset)

            act = 'no_exit'
            out_f = os.path.join('out_backprops', now_string())
            action_map = {'0': 'sel_img',
                          '1': 'set_mask',
                          '2': "gen_image",
                          '3': 'add_gen_to_dataset',
                          '4': 'flush_dataset',
                          '5': 'do_backprop',
                          '6': 'exit', '7': 'sel_cam',
                          '8': 'sel_gen',
                          '9' : 'save_mask',
                          '10' : 'load_mask'}

            while act != 'exit':
                try:
                    act = action_map.setdefault(
                        input("Accion? {0}".format(sorted(action_map.items()))),
                        '')

                    if act == 'sel_img':
                        ind_sel = input("Index ?")
                        img, label, all_cams, scores, r_label = get_img_cam(
                            ind_sel, base_dataset, model, ind_sel)
                        current_ind = ind_sel
                        current_img = img[0]
                        current_label = label
                        current_cams = all_cams
                    elif act == 'sel_cam':
                        selected_cam = int(input("Cam index ?"))

                    elif act == 'save_mask':
                        import pickle
                        out_path=os.path.join('./config_files/mask_files/mask_from_gen_{0}.pkl'.format(now_string()))
                        with open(out_path,'wb') as f:
                            pickle.dump(current_mask,f)
                        print("Mask saved to {0}".format(out_path))

                    elif act == 'load_mask':
                        import pickle
                        path_to_mask = input("Mask path?")
                        with open(path_to_mask,'rb') as f:
                            current_mask=pickle.load(f)
                        print("Mask loaded  {0}".format(path_to_mask))

                    elif act == 'sel_gen':
                        selected_key_gen = input("Gen key?")
                        gen_model = get_generator_from_key(selected_key_gen,
                                                           dataset=base_dataset)

                    elif act == 'set_mask':
                        cam_for_mask = np.squeeze(current_cams[selected_cam])
                        current_img = np.squeeze(current_img)
                        print(cam_for_mask.shape)
                        print(current_img.shape)
                        current_mask = sel_mask(cam_for_mask, current_img)

                    elif act == 'gen_image':
                        gen_images = gen_model.generate_img_mask(current_img,
                                                                 current_mask)
                        plt.figure()
                        plt.title('Original')
                        plt.imshow(current_img.squeeze())

                        for index_gen, gen_img in enumerate(gen_images):
                            plt.figure()
                            plt.title('gen_{0}'.format(index_gen))
                            plt.imshow(gen_img.squeeze())
                        plt.show()

                    elif act == 'add_gen_to_dataset':
                        for g_img in gen_images:
                            n_index = "gen_id__{0}__bindex__{1}".format(
                                current_ind, gens)
                            index_list.append(n_index)
                            st_gen_index = n_index if (
                                        st_gen_index is None) else st_gen_index
                            gens += 1
                            img_list.append(g_img)
                            label_list.append(current_label)

                    elif act == 'flush_dataset':
                        # import ipdb
                        # ipdb.set_trace()
                        index_list = np.array(index_list)
                        img_list = np.array(img_list)
                        label_list = np.array(label_list)

                        dataset_one_use.prepare_dataset(index_list, img_list,
                                                        label_list)
                        dataset_one_use.show_current()

                        index_list = []
                        img_list = []
                        label_list = []

                    elif act == 'do_backprop':

                        if backprops == 0:
                            eval_current_model('real', model, base_dataset,
                                               current_ind, backprops, out_f,
                                               model.current_log, selected_cam,
                                               eval=True)
                            eval_current_model('gen', model, dataset_one_use,
                                               st_gen_index, backprops, out_f,
                                               model.current_log, selected_cam)
                        backprops += 1
                        model.train(train_file_used=config_path,
                                    save_model=False, eval=False)
                        eval_current_model('real', model, base_dataset,
                                           current_ind, backprops, out_f,
                                           model.current_log, selected_cam,
                                           eval=True)
                        eval_current_model('gen', model, dataset_one_use,
                                           st_gen_index, backprops, out_f,
                                           model.current_log, selected_cam)

                    pass
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    print("Exception try again")





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