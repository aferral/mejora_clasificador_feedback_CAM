import os
import json

from utils import imshow_util, get_img_cam_index
from select_tool import ROOT_DIR
import numpy as np
import datetime
import cv2
import pickle

from select_tool.config_data import dataset_obj_dict, model_obj_dict, m_config

config_folder = os.path.join(ROOT_DIR,'config_files','train_result')
masks_folder = os.path.join(ROOT_DIR,'config_files','mask_files')

os.makedirs(os.path.join(ROOT_DIR,'config_files'),exist_ok=True)
os.makedirs(config_folder,exist_ok=True)
os.makedirs(masks_folder,exist_ok=True)




def create_empty_mask_file(mask_folder,dataset_key,classifier_key,name=None):
    now_string = datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
    if name:
        out_path = os.path.join(mask_folder,"mask_{3}_{0}_{1}_{2}.pkl".format(dataset_key,classifier_key,now_string,name))
    else:
        out_path = os.path.join(mask_folder,"mask_dummy_{0}_{1}_{2}.pkl".format(dataset_key, classifier_key, now_string))

    empty_mask_object = {"dataset_key" : dataset_key,'classifier_key' : classifier_key, 'masks' : {}}

    with open(out_path,'wb') as f:
        pickle.dump(empty_mask_object,f)

    return os.path.relpath(out_path, start=ROOT_DIR)

def get_config_file_list():
    files = os.listdir(config_folder)
    json_files = list(filter(lambda x : os.path.splitext(x)[-1] == '.json', files))
    return json_files


class model_manager_obj:

    def load_from_file(self,path):
        with open(path, 'r') as f:
            config_d = json.load(f)

        self.current_config_file = path

        # Check if json has all the keys
        assert (all([var in config_d.keys() for var in m_config.keys()]))

        # open train_file
        with open(config_d['train_file_used'],'r') as f:
            train_f_d = json.load(f)

        self.current_train_file = config_d['train_file_used']
        self.saved_model_path = config_d['model_load_path']
        self.mask_file_list = config_d['mask_files']

        self.dataset_key = train_f_d['dataset_key']
        self.classifier_key = train_f_d['model_key']
        self.dataset_params = train_f_d['dataset_params']
        self.classifier_params = train_f_d['model_params']


        # get dataset object
        dataset_obj = dataset_obj_dict[self.dataset_key](1, 1, **self.dataset_params)
        # get classifier object
        classifier = model_obj_dict[self.classifier_key](dataset_obj, **self.classifier_params)

        self.dataset_obj = dataset_obj
        self.classifier = classifier

        if len(self.mask_file_list) == 0:
            print("No mask files creating an empty mask file")
            new_mask_file = create_empty_mask_file(masks_folder, self.dataset_key, self.classifier_key)
            self.mask_file_list.append(new_mask_file)
            self.update_config_file()

        self.index_list = dataset_obj.get_index_list()
        self.current_index = self.index_list[0]
        self.current_mask_file = self.mask_file_list[0]
        self.current_mask_index_map = None

        self.load_mask_index_list()

    def __init__(self,path):
        self.load_from_file(path)


    def update_config_file(self):
        d={}
        d['train_file_used'] = self.current_train_file
        d['mask_files'] = self.mask_file_list
        d['model_load_path'] = self.saved_model_path

        with open(self.current_config_file, 'w') as f:
            json.dump(d,f)

    def load_mask_index_list(self):

        with open(self.current_mask_file, 'rb') as f:
            d=pickle.load(f)
        d_key = d['dataset_key']
        c_key = d['classifier_key']
        assert(self.dataset_key == d_key),'Saved mask has different dataset key. Expected {0} got {1}'.format(self.dataset_key,d_key)
        assert(self.classifier_key == c_key),'Saved mask has different classifier key. Expected {0} got {1}'.format(self.classifier_key,c_key)

        self.current_mask_index_map = d['masks']

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
        self.classifier.__exit__( exc_type, exc_val, exc_tb) # close already open classifier

    def change_model(self,path):

        self.classifier.close() # classifier is an exit stack so it can handle session close

        # open new classifier
        self.load_from_file(path)
        self.classifier.__enter__()

    def get_n_classes(self):
        return self.dataset_obj.shape_target[0]

    def get_current_index(self):
        return self.current_index

    def set_current_index(self, index):
        # check if index existe in index_list
        assert (index in self.index_list)
        self.current_index = index

    def get_index_list(self):
        return self.index_list

    def get_current_mask_index_list(self):
        return self.current_mask_index_map

    def get_current_mask_file(self):
        return self.current_mask_file

    def set_mask_file(self, mask_file):
        # check if mask_file in list
        assert (mask_file in self.mask_file_list)
        self.current_mask_file = mask_file
        self.load_mask_index_list()

    def get_mask_file_list(self):
        return self.mask_file_list

    def add_mask_file(self, name):
        new_mask_file = create_empty_mask_file(masks_folder, self.dataset_key, self.classifier_key,name=name)
        self.mask_file_list.append(new_mask_file)
        self.update_config_file()


    def get_img_cam_index(self, index):
        # check if index existe in index_list
        print("Getting {0}".format(index))
        assert (index in self.index_list)
        return get_img_cam_index(self.dataset_obj, self.classifier, index)

    def update_mask_file(self):
        print("Updating mask file {0}".format(self.current_mask_file))

        t={"dataset_key": self.dataset_key, 'classifier_key': self.classifier_key, 'masks': self.current_mask_index_map}
        with open(self.current_mask_file,'wb') as f:
            pickle.dump(t,f)

    def get_mask(self, index):
        shape_2d = self.dataset_obj.vis_shape()[0:2]

        if index in self.current_mask_index_map:
            return self.current_mask_index_map[index].reshape(shape_2d).copy()
        else:
            print("Creating mask for index {0}".format(index))

            new_mask = np.zeros(shape_2d).astype(np.bool)
            self.current_mask_index_map[index] = new_mask
            # update file
            self.update_mask_file()

        return self.current_mask_index_map[index].reshape(shape_2d).copy()

    def set_mask(self, index, mask):
        self.current_mask_index_map[index] = mask
        self.update_mask_file()

if __name__ == '__main__':
    with model_manager_obj('model_files/config_files/a.json') as model_manager:
        model_manager.get_mask_file_list()
