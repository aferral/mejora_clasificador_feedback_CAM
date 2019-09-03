import pickle

import numpy as np
from exp_backprops import do_train_config
import tensorflow as tf

"""
   # Exp shurikens

    

"""

# 20 for v2 with only a few mask, 31 for all

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1))

config_file = './config_files/train_result/Xray_dataset__imagenet_classifier_cam_loss_V2__21_Apr_2019__11_45.json'
exp_params_dict = {}
exp_params_dict['indexs_string'] = "train_16,train_513,train_632,train_479"
exp_params_dict['n_iterations'] = 100
exp_params_dict['iter_till_insert'] = 1
exp_params_dict['gens_per_original'] = 1
exp_params_dict['skip_insert'] = False
exp_params_dict['batch_size_exp'] = 50
exp_params_dict['exp_list'] = [10] # 9 solamente algunas mascasras , 10 todas las mascaras
exp_params_dict['base_name'] = 'xray_V2_f'
exp_params_dict['dropout_k'] = 20
exp_params_dict['mask_file_path_map'] = "config_files/mask_files/mask_dummy_xray_imagenet_cam_loss_V2_2019-04-21__12:45:11.pkl"

# temp list file
with open(exp_params_dict['mask_file_path_map'],'rb') as f:
    mask_dict = pickle.load(f)
with open('temp_list.txt','w') as f:
    f.write('\n'.join(mask_dict['masks'].keys()))

exp_params_dict['plot_masks'] = True
exp_params_dict['missclass_index_path'] = 'temp_list.txt'
exp_params_dict['add_summary'] = True

exp_params_dict['param_a'] = 8 # 4 0.4
exp_params_dict['param_b'] = 3 # 0.7
exp_params_dict['tf_config'] = config

res_path = do_train_config(config_file, **exp_params_dict)
