#!/usr/bin/python
import json
import http.server
import socketserver
from io import BytesIO
import os

import cv2
from scipy.misc import imsave
from select_tool.config_data import dataset_obj_dict
from utils import parse_config_recur
import pickle

import numpy as np

mask_image_file = "dataset_acts_2018-Dec-11--22:39.pkl"

PORT = 8000

config_file = './config_files/train_files/gen_imagenet_subset_09_oct_074.json'
data_config=parse_config_recur(config_file)

d_k = data_config['dataset_key']
d_p = data_config['dataset_params']
batch_size = 20
epochs = 1
dataset_class = dataset_obj_dict[d_k]
base_dataset = dataset_class(epochs, batch_size, **d_p)


# Open mean_vectors dataset
if mask_image_file:
    with open(mask_image_file, 'rb') as f:
        out = pickle.load(f)
    component_mask = out['comp_mask']


class myHandler(http.server.BaseHTTPRequestHandler):

    # Handler for the GET requests
    def do_GET(self):
        print("Received {0}".format(self.path))


        index=self.path[1:]

        if index == 'test':
            index = base_dataset.get_index_list()[0]
            res = base_dataset.get_train_image_at(index, strict=True)
        else:

            if mask_image_file:
                ind_img,ind_comp = index.split('--')
                res = base_dataset.get_train_image_at(ind_img, strict=True)


        if res:
            img_res = res[0][0]

            if mask_image_file:
                mask = component_mask[(ind_img,int(ind_comp))]
                if img_res.shape[0:-1] != mask.shape:
                    mask = cv2.resize(mask.astype(np.float32),img_res.shape[0:-1]).astype(np.bool)
                img_res[np.bitwise_not(mask)] = 0

            fname = 'tmp.png'

            imsave(fname, img_res)

            file_s=open(fname,'rb')
            mimetype = 'image/png'
            self.send_response(200)
            self.send_header('Content-type', mimetype)
            self.end_headers()
            self.wfile.write(file_s.read())
            file_s.close()
            os.remove(fname)

        else:
            print("NOT found in dataset {0}".format(index))
            self.send_error(404, 'File Not Found: %s' % self.path)


httpd=socketserver.TCPServer(("", PORT), myHandler)
try:
    print("serving at port", PORT)
    httpd.serve_forever()
except Exception:
    httpd.shutdown()


