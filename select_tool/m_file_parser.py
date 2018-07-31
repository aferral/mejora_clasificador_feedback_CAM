import os
import json
from classification_models.classification_model import CWR_classifier, \
    Abstract_model, imshow_util
from datasets.cwr_dataset import CWR_Dataset
from datasets.dataset import Dataset
from select_tool import ROOT_DIR
import numpy as np
import datetime
import cv2
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

        self.dataset_key = config_d['dataset_key']
        self.classifier_key = config_d['model_key']
        self.dataset_params = config_d['dataset_params']
        self.classifier_params = config_d['model_params']
        self.saved_model_path = config_d['model_load_path']
        self.mask_file_list = config_d['mask_files']

        # get dataset object
        dataset_obj = dataset_obj_dict[self.dataset_key](1, 1, **self.dataset_params)
        # get classifier object
        classifier = model_obj_dict[self.classifier_key](dataset_obj, **self.classifier_params)

        self.dataset_obj = dataset_obj
        self.classifier = classifier

        if len(self.mask_file_list) == 0:
            print("No mask files creating an empty mask file")
            new_mask_file = create_empty_mask_file(masks_folder, config_d['dataset_key'], config_d['model_key'])
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
        d['dataset_key'] = self.dataset_key
        d['model_key'] = self.classifier_key
        d['mask_files'] = self.mask_file_list
        d['dataset_params'] = self.dataset_params
        d['model_params'] = self.classifier_params
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

        self.classifier.close() # classifier is an exit stack so close handle session close

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

        batch_x = self.dataset_obj.get_train_image_at(index)
        test_image = batch_x[0]
        img = imshow_util(test_image.reshape(self.dataset_obj.vis_shape()), self.dataset_obj.get_data_range())

        image_processed, prediction, cmaps = self.classifier.visualize(test_image)

        img_cam = imshow_util(image_processed.reshape(self.dataset_obj.vis_shape()), self.dataset_obj.get_data_range())

        all_cams =[]
        for i in range(cmaps.shape[0]):
            im = (cmaps[i]*255).astype(np.uint8)
            t=cv2.resize(im, (img.shape[0], img.shape[1]))
            all_cams.append(t)

        return (img * 255).astype(np.uint8),all_cams,prediction

    def update_mask_file(self):
        print("Updating mask file {0}".format(self.current_mask_file))

        t={"dataset_key": self.dataset_key, 'classifier_key': self.classifier_key, 'masks': self.current_mask_index_map}
        with open(self.current_mask_file,'wb') as f:
            pickle.dump(t,f)

    def get_mask(self, index):

        if index in self.current_mask_index_map:
            return self.current_mask_index_map[index].reshape(self.dataset_obj.vis_shape()).copy()
        else:
            print("Creating mask for index {0}".format(index))

            new_mask = np.zeros(self.dataset_obj.shape).astype(np.bool)
            self.current_mask_index_map[index] = new_mask
            # update file
            self.update_mask_file()

        return self.current_mask_index_map[index].reshape(self.dataset_obj.vis_shape()).copy()

    def set_mask(self, index, mask):
        self.current_mask_index_map[index] = mask
        self.update_mask_file()

if __name__ == '__main__':
    with model_manager_obj('model_files/config_files/a.json') as model_manager:
        model_manager.get_mask_file_list()
