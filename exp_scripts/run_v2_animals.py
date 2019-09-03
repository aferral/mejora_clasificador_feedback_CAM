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
    params_cand_a = [0.4, 0.2, 0.8]
    params_cand_b = [0.7,0,5,1.2]

    candidatos = [ (0.4,0.7), (0.4,0.5), (0.4,1.2),
                   (0.2,0.7), (0.2,0.5), (0.2,1.2) ,
                   (0.8,0.7), (0.8,0.5), (0.8,1.2) ]

    print(sorted(candidatos))


    result_paths = {'exp_path' : [],'param_a' : [], 'param_b' : []}
    for ind,(pa,pb) in enumerate(candidatos):
        print('-----------------------------------------------')
        print('-----------------------------------------------')
        print('Doing EXP: {0} with params: {1}'.format(ind,(pa,pb)))
        print('-----------------------------------------------')
        print('-----------------------------------------------')


        config_file = './config_files/train_files/gen_imagenet_subset_09_oct_074.json'
        exp_params_dict = {}
        exp_params_dict['indexs_string'] = 'n02114548_10505.JPEG,n02120079_9808.JPEG,n02114548_11513.JPEG,n02120079_4409.JPEG,n02114548_5207.JPEG'
        exp_params_dict['n_iterations'] = 100
        exp_params_dict['iter_till_insert'] = 1
        exp_params_dict['gens_per_original'] = 1
        exp_params_dict['skip_insert'] = False
        exp_params_dict['batch_size_exp'] = 50
        exp_params_dict['exp_list'] = [9]
        exp_params_dict['base_name'] = 'loss_v2_few_{0}'.format(ind)
        exp_params_dict['dropout_k'] = 20
        exp_params_dict['mask_file_path_map'] =  "sel_mask_exp_3.pkl"
        exp_params_dict['plot_masks'] = True
        exp_params_dict['add_summary'] = True
        exp_params_dict['lambda_value'] = best_lambda

        exp_params_dict['param_a'] = pa
        exp_params_dict['param_b'] = pb

        res_path = do_train_config(config_file, **exp_params_dict)
        result_paths['exp_path'].append(res_path)
        result_paths['param_a'].append(pa)
        result_paths['param_b'].append(pb)


    with open("resultados_params_V2_exp.json",'w') as f:
        json.dump(result_paths,f)


import matplotlib.pyplot as plt

with open("resultados_params_V2_exp.json",'r') as f:
    result_paths=json.load(f)

# plot lambdas vs accuracy
pa_r = []
pb_r = []
accs = []
val_acc_col = 1
for i in range(len(result_paths['exp_path'])):
    path_resx = result_paths['exp_path'][i]
    pa_r.append(result_paths['param_a'][i])
    pb_r.append(result_paths['param_b'][i])

    path_accs = os.path.join(path_resx, 'accuracy_simple.csv')
    data=pd.read_csv(path_accs,header=None)
    acc_last_it = data.iloc[-1][val_acc_col]
    accs.append(acc_last_it)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pa_r, pb_r, accs, zdir='z', s=20)
ax.set_xlabel('Param a')
ax.set_ylabel('Param b')
plt.show()