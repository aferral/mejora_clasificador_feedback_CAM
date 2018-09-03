import numpy as np
import tensorflow as tf
import neuralgym as ng
import os
import cv2
import json
import pickle
from datasets.dataset import Dataset
from  datetime import datetime

from datasets.imagenet_data import Imagenet_Dataset

import sys
path_m = os.path.abspath('../genm')
print(path_m)
sys.path.append(path_m)
from inpaint_model import InpaintCAModel


class Abstract_generator:


    def generate(self,dataset_object : Dataset,index_to_use,mask_file):

        out_folder = 'temp_gen' #todo add mask name , dataset_name, gen_name, datelocal
        os.makedirs(out_folder,exist_ok=True)

        # open masks pickle
        with open(mask_file,'rb') as f:
            mask_dict = pickle.load(f)

        for ind in index_to_use:
            mask = mask_dict['masks'][ind]
            img = dataset_object.get_train_image_at(ind)[0][0] # returns (img,index) , img = [batch,w,h,c]

            result = self.generate_img_mask(img,mask)

            cv2.imwrite(os.path.join(out_folder, '{0}__mask.png'.format(ind)), mask.astype(np.uint8)*255)

            for ind_out,elem in enumerate(result):
                cv2.imwrite(os.path.join(out_folder,'{0}__{1}.png'.format(ind,ind_out)),elem)

        exp_json = {'date' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'dataset' : str(dataset_object.__class__.__name__),
                    'mask_file' : str(mask_file),
                    'generator' : str(self.__class__.__name__)
                    }
        with open(os.path.join(out_folder,'exp_details.json'),'w') as f:
            json.dump(exp_json,f)

        pass

    def generate_img_mask(self,img,mask):
        raise NotImplementedError()

class yu2018generative(Abstract_generator):
    def __init__(self):
        self.virtal_env_source_path = 'venv/bin/activate'
        self.generative_model_path = '../genm/model_logs/release_imagenet_256'

    def generate_img_mask(self,image,mask):

        if len(image.shape) == 3:
            mask =np.repeat(mask[:,:,np.newaxis],image.shape[2],axis=2)
        mask = mask.astype(np.uint8) * 255

        ng.get_gpus(1)

        model = InpaintCAModel()

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 8
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(self.generative_model_path, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            result = sess.run(output)
            out_img = result[0][:, :, ::-1]
        return [out_img]

if __name__ == '__main__':
    t=yu2018generative()

    index_list = ['n02423022_5592.JPEG']
    d_obj = Imagenet_Dataset(1, 1,data_folder = "./temp/imagenet_subset")
    mask_file = 'model_files/mask_files/mask_dummy_Imagenet_subset_vgg_16_batchnorm_2018-08-01__11:14:41.pkl'
    t.generate(d_obj,index_list,mask_file)

    pass