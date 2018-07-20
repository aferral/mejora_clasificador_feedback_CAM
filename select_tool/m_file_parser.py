import os
import json
from classification_models.classification_model import CWR_classifier, \
    Abstract_model
from datasets.cwr_dataset import CWR_Dataset
from datasets.dataset import Dataset
from select_tool import ROOT_DIR
import numpy as np


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


def get_config_file_list():
    files = os.listdir(config_folder)
    json_files = list(filter(lambda x : os.path.splitext(x)[-1] == '.json', files))
    return json_files


class selected_model_obj:
    def __init__(self,path):
        with open(path, 'r') as f:
            config_d = json.load(f)

        # Check if json has all the keys

        # Check default value in

        # get dataset object
        dataset_params = config_d['dataset_params']
        dataset_obj = dataset_obj_dict[config_d['dataset_key']](1, 1,
                                                                **dataset_params)

        # get classifier object
        classifier_params = config_d['model_params']
        classifier = model_obj_dict[config_d['model_key']](dataset_obj,
                                                           **classifier_params)

        # load train weights if has model_load_path

        # load mask files
        all_mask_files = config_d['mask_files']


        all_indexs = dataset_obj.get_index_list()
        assert (len(all_indexs) > 0)
        assert (len(all_mask_files) > 0)

        self.dataset_obj = dataset_obj
        self.classifier = classifier

        self.current_index = all_indexs[0]
        self.index_list = all_indexs

        self.mask_file_list = all_mask_files
        self.current_mask_file = all_mask_files[0]
        self.current_mask_list = []

    def get_current_index(self):
        return self.current_index

    def set_current_index(self, index):
        # check if index existe in index_list
        assert (index in self.index_list)
        self.current_index = index

    def get_index_list(self):
        return self.index_list

    def get_mask_list(self):
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


parse_config_file('model_files/config_files/a.json')
