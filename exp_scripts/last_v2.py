import numpy as np
from exp_backprops import do_train_config
import os
import pandas as pd
import json
import pickle

# divide mask amount
best_lambda = 1.0/(14*14*64)*np.power(0.2,2)


config_file = './config_files/train_files/gen_imagenet_subset_09_oct_074.json'
exp_params_dict = {}
exp_params_dict['indexs_string'] = 'n02114548_10505.JPEG,n02120079_9808.JPEG,n02114548_11513.JPEG,n02120079_4409.JPEG,n02114548_5207.JPEG'
exp_params_dict['n_iterations'] = 100
exp_params_dict['iter_till_insert'] = 1
exp_params_dict['gens_per_original'] = 1
exp_params_dict['skip_insert'] = False
exp_params_dict['batch_size_exp'] = 50
exp_params_dict['exp_list'] = [9]
exp_params_dict['base_name'] = 'loss_v2_few_{0}'.format('FIN')
exp_params_dict['dropout_k'] = 20
exp_params_dict['mask_file_path_map'] =  "sel_mask_exp_3.pkl"
exp_params_dict['plot_masks'] = True
exp_params_dict['add_summary'] = True
exp_params_dict['lambda_value'] = best_lambda

exp_params_dict['param_a'] = 0.4
exp_params_dict['param_b'] = 0.7

res_path = do_train_config(config_file, **exp_params_dict)
