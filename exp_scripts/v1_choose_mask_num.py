import numpy as np
from exp_backprops import do_train_config
import os
import pandas as pd
import json
import pickle

redo = False

# divide mask amount
best_lambda = 1.0/(14*14*64)*np.power(0.2,2)


if redo:
    n_masks_to_use = [5,10,20,40,80,150,220,250,300]
    print(sorted(n_masks_to_use))


    all_masks_path = './config_files/mask_files/all_filtered_masks.pkl'
    with open(all_masks_path,'rb') as f:
        all_mask_dict = pickle.load(f)
    mask_keys_sorted = list(sorted(all_mask_dict['masks']))



    all_inds_list = ''

    result_paths = {'exp_path' : [],'n_masks' : []}
    for ind,nmasks in enumerate(n_masks_to_use):
        print('-----------------------------------------------')
        print('-----------------------------------------------')
        print('Doing EXP: {0} with nmasks: {1}'.format(ind,nmasks))
        print('-----------------------------------------------')
        print('-----------------------------------------------')

        #  create file with lx number of masks
        #  limit indexs list too
        temp_mask_dict = all_mask_dict.copy()
        selected_indexs = mask_keys_sorted[0: nmasks]
        temp_mask_dict['masks'] = {}
        temp_mask_dict['masks'] = {k: all_mask_dict['masks'][k] for k in selected_indexs}
        with open('temp_mask_file.pkl', 'wb') as f:
            pickle.dump(temp_mask_dict, f)
        with open('temp_inds_list.txt', 'w') as f:
            f.write('\n'.join(selected_indexs))


        config_file = './config_files/train_files/gen_imagenet_subset_09_oct_074.json'
        exp_params_dict = {}
        exp_params_dict['indexs_string'] = 'n02114548_10505.JPEG,n02120079_9808.JPEG,n02114548_11513.JPEG,n02120079_4409.JPEG,n02114548_5207.JPEG'
        exp_params_dict['n_iterations'] = 100
        exp_params_dict['iter_till_insert'] = 1
        exp_params_dict['gens_per_original'] = 1
        exp_params_dict['skip_insert'] = False
        exp_params_dict['batch_size_exp'] = 50
        exp_params_dict['exp_list'] = [13]
        exp_params_dict['base_name'] = 'loss_v1_allmasks_iterative_{0}'.format(ind)
        exp_params_dict['dropout_k'] = 20
        exp_params_dict['mask_file_path_map'] = 'temp_mask_file.pkl'
        exp_params_dict['plot_masks'] = True
        exp_params_dict['missclass_index_path'] = 'temp_inds_list.txt'
        exp_params_dict['add_summary'] = True
        exp_params_dict['lambda_value'] = best_lambda

        res_path = do_train_config(config_file, **exp_params_dict)
        result_paths['exp_path'].append(res_path)
        result_paths['n_masks'].append(nmasks)

        os.remove('temp_mask_file.pkl')
        os.remove('temp_inds_list.txt')

    with open("resultados_mask_incremental_V1_exp.json",'w') as f:
        json.dump(result_paths,f)


import matplotlib.pyplot as plt

with open("resultados_mask_incremental_V1_exp.json",'r') as f:
    result_paths=json.load(f)

# plot lambdas vs accuracy
n_masks = []
accs = []
val_acc_col = 1
for i in range(len(result_paths['exp_path'])):
    path_resx = result_paths['exp_path'][i]
    lambda_x = result_paths['n_masks'][i]

    path_accs = os.path.join(path_resx, 'accuracy_simple.csv')
    data=pd.read_csv(path_accs,header=None)
    acc_last_it = data.iloc[-1][val_acc_col]

    n_masks.append(lambda_x)
    accs.append(acc_last_it)

with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.plot(n_masks,accs,'*--')
    ax.set_xlabel('Cantidad mascaras')
    ax.set_ylabel('Accuracy validacion')
    ax.grid()
    plt.savefig('seleccion_mascaras.png', dpi=100)