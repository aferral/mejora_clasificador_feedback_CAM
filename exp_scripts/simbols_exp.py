import numpy as np
from exp_backprops import do_train_config
import tensorflow as tf
"""
    indices simple experiment
    
    './config_files/train_result/Simple_figures_dataset__simple_classifier__10_Feb_2019__19_51.json' 
    'epsilon_39.png,xi_22.png,zeta_49.png'
    './config_files/mask_files/mask_dummy_simple_figures_simple_model_2019-02-10__19:58:50.pkl'
    
    
    'simple_index_list.txt'
    
    config_file = './config_files/train_result/Simple_figures_dataset__simple_classifier__10_Feb_2019__19_51.json'
    exp_params_dict['indexs_string'] =  'epsilon_39.png,xi_22.png,zeta_49.png'
    exp_params_dict['exp_list'] = [6]
    exp_params_dict['mask_file_path_map'] =   'config_files/mask_files/mask_dummy_simple_figures_simple_model_2019-02-10__19:58:50.pkl'
    exp_params_dict['missclass_index_path'] = 'simple_index_list.txt'
    
    
    # TOMA 2 de simple experiment
    
    'model/simple_classifier/10_Mar_2019__02_17'
    ./config_files/train_result/Simple_figures_dataset__simple_classifier__10_Mar_2019__02_18.json
    'epsilon_39.png,xi_22.png,zeta_49.png'
    
    config_file = './config_files/train_result/Simple_figures_dataset__simple_classifier__10_Mar_2019__02_18.json'
    exp_params_dict = {}
    exp_params_dict['indexs_string'] =  'epsilon_39.png,xi_22.png,zeta_49.png'
    exp_params_dict['exp_list'] = [6]
    exp_params_dict['mask_file_path_map'] =   'config_files/mask_files/mask_dummy_simple_figures_simple_model_2019-02-10__19:58:50.pkl'
    exp_params_dict['missclass_index_path'] = 'simple_index_list.txt'
    
    
    # Exp simple V2 (ruido en ambos lados)
    
    ./config_files/train_result/Simple_figures_dataset__simple_classifier__29_Mar_2019__17_05.json
    config_file = './config_files/train_result/Simple_figures_dataset__simple_classifier__29_Mar_2019__17_05.json'
    exp_params_dict['indexs_string'] = 'epsilon_46.png,xi_22.png,zeta_49.png'
    exp_params_dict['exp_list'] = [31]
    exp_params_dict[
        'mask_file_path_map'] = 'config_files/mask_files/mask_dummy_simple_figures_simple_model_2019-02-10__19:58:50.pkl'
    exp_params_dict['missclass_index_path'] = 'simple_index_list.txt'
    
    # Exp simple V3 (ruido ajustado para un lado)
    config_file = './config_files/train_result/Simple_figures_dataset__simple_classifier__20_Apr_2019__21_48.json'
    exp_params_dict['exp_list'] = [10]
    exp_params_dict['mask_file_path_map'] = 'config_files/mask_files/mask_simple_fig_reducido.pkl'
    exp_params_dict['missclass_index_path'] = 'lista_index_simple_reducido.txt'

"""
# 20 for v2 with only a few mask, 31 for all

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333))


config_file = './config_files/train_result/Simple_figures_dataset__simple_classifier__20_Apr_2019__21_48.json'
exp_params_dict = {}
exp_params_dict['indexs_string'] = 'epsilon_46.png,xi_22.png,zeta_49.png'
exp_params_dict['n_iterations'] = 100
exp_params_dict['iter_till_insert'] = 1
exp_params_dict['gens_per_original'] = 1
exp_params_dict['skip_insert'] = False
exp_params_dict['batch_size_exp'] = 50
exp_params_dict['exp_list'] = [6]
exp_params_dict['base_name'] = 'loss_V2_SIMPLE_cont_{0}'.format('FIN')
exp_params_dict['dropout_k'] = 20
exp_params_dict['mask_file_path_map'] = 'config_files/mask_files/mask_simple_fig_reducido.pkl'
exp_params_dict['missclass_index_path'] = 'lista_index_simple_reducido.txt'
exp_params_dict['plot_masks'] = True
exp_params_dict['add_summary'] = True
exp_params_dict['param_a'] = 0.4
exp_params_dict['param_b'] = 0.7
exp_params_dict['tf_config'] = config



res_path = do_train_config(config_file, **exp_params_dict)
