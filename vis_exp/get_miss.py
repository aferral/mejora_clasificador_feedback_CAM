import cv2
import tensorflow as tf
import numpy as np
from skimage import measure
from sklearn.cluster import MiniBatchKMeans
from classification_models.imagenet_subset_cam_loss import \
    imagenet_classifier_cam_loss
from select_tool.config_data import model_obj_dict, dataset_obj_dict
from utils import parse_config_recur
from skimage.filters import threshold_otsu, rank
from scipy.sparse.csgraph import connected_components
import pickle
from datetime import datetime
from vis_exp.vis_bokeh import plotBlokeh


import argparse

argparser =argparse.ArgumentParser()
argparser.add_argument('config_file',help='Train file used ')


args = argparser.parse_args()


config_file = args.config_file



data_config = parse_config_recur(config_file)
t_params = data_config['train_params']
m_k = data_config['model_key']
m_p = data_config['model_params']
d_k = data_config['dataset_key']
d_p = data_config['dataset_params']

if data_config['model_load_path_at_train_file'] is not None:
    model_load_path = data_config['model_load_path_at_train_file']
    print("Using model load path from TRAIN_FILE {0}".format(model_load_path))
else:
    model_load_path = data_config['model_load_path_train_result']
    print("Using model load path from SELECT_FILE {0}".format(model_load_path))

batch_size = t_params['b_size'] if 'b_size' in t_params else 20
epochs = t_params['epochs'] if 'epochs' in t_params else 1


# open dataset
model_class = imagenet_classifier_cam_loss
dataset_class = dataset_obj_dict[d_k]
base_dataset = dataset_class(epochs, batch_size, **d_p)  # type: Dataset


# iterate train set

with model_class(base_dataset,**m_p) as model:  # this also save the train_result
    model.load(model_load_path)

    model.dataset.initialize_iterator_train(model.sess)

    counter=0
    all_indexs = []

    while True:
        try:
            fd = model.prepare_feed(is_train=False, debug=False)

            # extract activations in batch, CAM
            index_batch,soft_max_out,targets = model.sess.run([model.indexs,model.pred,model.targets],feed_dict=fd)


            miss_class = (soft_max_out.argmax(axis=1) != targets.argmax(axis=1))



            index_batch = index_batch[miss_class]
            all_indexs += index_batch.tolist()



            if counter % 3 == 0:
                print(counter)
            counter += 1

        except tf.errors.OutOfRangeError:
            log = 'break at {0}'.format(counter)
            break



# save component_mean_v, component_mask, mean_per_filter
with open("{0}_indexs_miss.pkl".format(d_k),'wb') as f:
    pickle.dump(all_indexs,f,-1)
