import numpy as np

import os
import cv2
import json
import pickle
from datasets.dataset import Dataset
from  datetime import datetime


from utils import now_string


def try_to_adjust_to_3c_mask(image, mask):
    n_mask = mask.copy()
    if len(image.shape) == 2 or (image.shape[2] == 1):
        image = np.stack([image[:, :].reshape(image.shape[0], image.shape[1]) for i in range(3)], axis=2)
        n_mask = np.stack(
            [n_mask[:, :].reshape(n_mask.shape[0], n_mask.shape[1]) for i in
             range(3)], axis=2)


    if len(image.shape) == 3 and (len(n_mask.shape) != 3):
        print("x")
        n_mask = np.repeat(n_mask[:, :, np.newaxis], image.shape[2], axis=2)
    n_mask = n_mask.astype(np.uint8) * 255

    return n_mask,image

def try_to_adjust_to_shape_mask(image, mask):
    n_mask = mask.copy()

    # image(a,b) image(a,b,1) reshape mask, img to (a,b,1)
    if len(image.shape) == 2 or (image.shape[2] == 1):
        n_mask = n_mask.reshape(image.shape[0],image.shape[1],1)
        image = image.reshape(image.shape[0],image.shape[1],1)
    elif len(image.shape) == 3:     # image(a,b,c) mask(a,b) repeat mask
        n_channels = image.shape[2]
        n_mask = np.stack(
            [n_mask[:, :].reshape(n_mask.shape[0], n_mask.shape[1]) for i in
             range(n_channels)], axis=2)

    n_mask = n_mask.astype(np.uint8) * 255

    return n_mask,image


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
            img = dataset_object.get_train_image_at(ind)[0][0] # returns (img,label) , img = [batch,w,h,c]

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
        """
        Generate image replacing masked parts of image
        :param img: Image WITHOUT preprocessing
        :param mask: binary mask of shape of image
        :return:
        """
        raise NotImplementedError()

