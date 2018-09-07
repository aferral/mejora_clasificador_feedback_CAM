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

from select_tool.config_data import dataset_obj_dict
from utils import now_string

path_m = os.path.abspath('../genm')
print(path_m)
sys.path.append(path_m)
from inpaint_model import InpaintCAModel


class Abstract_generator:


    def generate(self, dataset_object : Dataset, index_list, mask_file,select_path):

        d_name=str(dataset_object.__class__.__name__)
        gen_name=str(self.__class__.__name__)
        f_name = '{0}_{1}_{2}'.format(d_name,gen_name,now_string())
        out_folder = os.path.join('gen_images', f_name)
        os.makedirs(out_folder,exist_ok=True)

        gen_dict = {}

        # open masks pickle
        with open(mask_file,'rb') as f:
            mask_dict = pickle.load(f)

        for ind in index_list:
            mask = mask_dict['masks'][ind]
            img = dataset_object.get_train_image_at(ind)[0][0] # returns (img,index) , img = [batch,w,h,c]

            result = self.generate_img_mask(img,mask)

            cv2.imwrite(os.path.join(out_folder, '{0}__mask.png'.format(ind)), mask.astype(np.uint8)*255)

            gen_dict[ind] = []
            for ind_out,elem in enumerate(result):
                out_path_img = os.path.join(out_folder,'{0}__{1}.png'.format(ind,ind_out))
                cv2.imwrite(out_path_img,elem)
                gen_dict[ind].append(out_path_img)

        exp_json = {'date' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'dataset' : d_name,
                    'used_select' : select_path,
                    'index_map' : gen_dict,
                    'mask_file' : str(mask_file),
                    'generator' : gen_name
                    }
        print("Results in "+str(os.path.join(out_folder,'exp_details.json')))
        with open(os.path.join(out_folder,'exp_details.json'),'w') as f:
            json.dump(exp_json,f)


    def generate_img_mask(self,img,mask):
        raise NotImplementedError()

class yu2018generative(Abstract_generator):
    def __init__(self):
        self.virtal_env_source_path = 'venv/bin/activate'
        self.generative_model_path = '../genm/model_logs/release_imagenet_256'

    def generate_img_mask(self,image,mask):

        if len(image.shape) == 2 or (image.shape[2] == 1):
            image = np.stack([image[:,:].reshape(image.shape[0],image.shape[1]) for i in range(3)],axis=2)
            mask = np.stack([mask[:,:].reshape(mask.shape[0],mask.shape[1]) for i in range(3)],axis=2)

        if len(image.shape) == 3 and (len(mask.shape) != 3):
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

        tf.reset_default_graph()
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
    import argparse

    available_map = {'yu2018' : yu2018generative}

    parser = argparse.ArgumentParser(description='Execute generative')
    parser.add_argument('select_config_json', help='The config_json to train')
    parser.add_argument('method_name', help='Generative algorithm to use. Available: {0}'.format(list(available_map.keys())))

    args = parser.parse_args()
    config_path = args.select_config_json
    method_name = args.method_name

    with open(config_path,'r') as f:
        data_select = json.load(f)

    train_result_path = data_select['train_result_path']
    mask_file = data_select['mask_file']
    index_list = data_select['index_list']

    with open(train_result_path,'r') as f:
        data_t_r = json.load(f)
        path_train_file = data_t_r["train_file_used"]

        with open(path_train_file,'r') as f2:
            data_train_file = json.load(f2)

        d_k = data_train_file['dataset_key']
        d_p = data_train_file['dataset_params']


    t= available_map[method_name]()

    d_obj = dataset_obj_dict[d_k](1,1,**d_p)
    t.generate(d_obj,index_list,mask_file,config_path)

