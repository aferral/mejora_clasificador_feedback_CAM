import os
import json
from classification_models.classification_model import CWR_classifier, \
    Abstract_model
from datasets.cwr_dataset import CWR_Dataset
from datasets.dataset import Dataset
from select_tool import ROOT_DIR
import numpy as np
import datetime

import pickle


config_folder = os.path.join(ROOT_DIR,'model_files','config_files')
masks_folder = os.path.join(ROOT_DIR,'model_files','mask_files')

os.makedirs(os.path.join(ROOT_DIR,'model_files'),exist_ok=True)
os.makedirs(config_folder,exist_ok=True)
os.makedirs(masks_folder,exist_ok=True)


dataset_obj_dict = {
    'CWR' : CWR_Dataset,

}

model_obj_dict = {
    'CWR_classifier': CWR_classifier,
}

default_keys = ['dataset_key','model_key','mask_files','dataset_params','model_params','model_load_path']
default_values = [None,None,[],{},{},None]
requiered = [True,True,False,False,False,False]


# DEFAULT CONFIG DICTIONARY
m_config = {
    'dataset_key' : 'CWR',
    'model_key' : 'CWR_classifier',
    'mask_files' : [],
    'dataset_params' : {},
    'model_params' : {},
    'model_load_path' : None
}


def create_empty_mask_file(mask_folder,dataset_key,classifier_key):
    now_string = datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
    out_path = os.path.join(mask_folder,"mask_dummy_{0}_{1}_{2}.pkl".format(dataset_key,classifier_key,now_string))

    empty_mask_object = {"dataset_key" : dataset_key,'classifier_key' : classifier_key, 'masks' : {}}

    with open(out_path,'wb') as f:
        pickle.dump(empty_mask_object,f)

    return out_path

def get_config_file_list():
    files = os.listdir(config_folder)
    json_files = list(filter(lambda x : os.path.splitext(x)[-1] == '.json', files))
    return json_files


class model_manager_obj:
    def __init__(self,path):
        with open(path, 'r') as f:
            config_d = json.load(f)

        # Check if json has all the keys
        assert(all([var in config_d.keys() for var in m_config.keys() ]))

        self.dataset_key = config_d['dataset_key']
        self.classifier_key = config_d['model_key']


        # get dataset object
        dataset_params = config_d['dataset_params']
        dataset_obj = dataset_obj_dict[self.dataset_key](1, 1,
                                                                **dataset_params)

        # get classifier object
        classifier_params = config_d['model_params']
        classifier = model_obj_dict[self.classifier_key](dataset_obj,
                                                           **classifier_params)


        # load mask files
        all_mask_files = config_d['mask_files']


        all_indexs = dataset_obj.get_index_list()
        assert (len(all_indexs) > 0)

        if len(all_mask_files) == 0 :
            print("No mask files creating an empty mask file")
            new_mask_file = create_empty_mask_file(masks_folder, config_d['dataset_key'], config_d['model_key'])
            all_mask_files.append(new_mask_file)

        self.dataset_obj = dataset_obj
        self.classifier = classifier
        self.saved_model_path = config_d['model_load_path']

        self.current_index = all_indexs[0]
        self.index_list = all_indexs

        self.mask_file_list = all_mask_files
        self.current_mask_file = all_mask_files[0]

        self.load_mask_index_list()

    def load_mask_index_list(self):

        with open(self.current_mask_file, 'rb') as f:
            d=pickle.load(f)
        d_key = d['dataset_key']
        c_key = d['classifier_key']
        assert(self.dataset_key == d_key),'Saved mask has different dataset key. Expected {0} got {1}'.format(self.dataset_key,d_key)
        assert(self.classifier_key == c_key),'Saved mask has different classifier key. Expected {0} got {1}'.format(self.classifier_key,c_key)

        self.current_mask_list = d['masks']

    def __enter__(self):
        self.classifier.__enter__()

        # start TF session
        # load train weights if has model_load_path
        if not (self.saved_model_path is None):
            self.classifier.load(self.saved_model_path)
        else:
            print("Model load path is None. Using initialized weights")

        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.classifier.__exit__( exc_type, exc_val, exc_tb)

    def change_model(self,path):
        pass

    def get_current_index(self):
        return self.current_index

    def set_current_index(self, index):
        # check if index existe in index_list
        assert (index in self.index_list)
        self.current_index = index

    def get_index_list(self):
        return self.index_list

    def get_current_mask_index_list(self):
        return self.current_mask_list

    def get_current_mask_file(self):
        return self.current_mask_file

    def set_mask_file(self, mask_file):
        # check if mask_file in list
        assert (mask_file in self.mask_file_list)
        self.current_mask_file = mask_file

    def get_mask_file_list(self):
        return self.mask_file_list

    def add_mask_file(self, name):
        self.mask_file_list.append(name)
        # todo crear achivo a disco


    def get_img_index(self, index):
        # check if index existe in index_list
        print(index)
        assert (index in self.index_list)
        return self.dataset_obj.get_index(index)

    def get_img_cam(self, index):
        # todo
        return (np.random.rand(200, 400, 3) * 255).astype(np.uint8)

    def get_mask(self, index):
        # todo asegurate de crearla en caso de que no exista.
        img_cam = (np.random.rand(200, 400, 3) * 255).astype(np.uint8)
        return np.zeros((img_cam.shape[0], img_cam.shape[1])).astype(np.bool)

    def set_mask(self, index, mask):
        # todo asegurate de crearla en caso de que no exista.
        return None

if __name__ == '__main__':
    with model_manager_obj('model_files/config_files/a.json') as model_manager:
        model_manager.get_mask_file_list()
