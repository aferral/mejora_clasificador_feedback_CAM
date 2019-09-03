import numpy as np
from exp_backprops import do_train_config
import os
import pandas as pd
import json

redo = False

if redo:
    l0=1.0/(14*14*64)
    lambdas_to_use = [l0]
    lambdas_to_use += [l0*np.power(0.2,i) for i in range(1,4)]
    lambdas_to_use += [l0*np.power(2,i) for i in range(1,4)]
    print(sorted(lambdas_to_use))

    result_paths = {'exp_path' : [],'lambda' : []}
    for ind,lx in enumerate(lambdas_to_use):
        print('-----------------------------------------------')
        print('-----------------------------------------------')
        print('Doing EXP: {0} with lambda {1}'.format(ind,lx))
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
        exp_params_dict['exp_list'] = [12]
        exp_params_dict['base_name'] = 'loss_v1_lambda_iterative_{0}'.format(ind)
        exp_params_dict['dropout_k'] = 20
        exp_params_dict['mask_file_path_map'] = "sel_mask_exp_3.pkl"
        exp_params_dict['plot_masks'] = True
        exp_params_dict['missclass_index_path'] = 'all_masks_indexs.txt'
        exp_params_dict['add_summary'] = True
        exp_params_dict['lambda_value'] = lx

        res_path = do_train_config(config_file, **exp_params_dict)
        result_paths['exp_path'].append(res_path)
        result_paths['lambda'].append(lx)
    with open("resultados_lambda_exp.json",'w') as f:
        json.dump(result_paths,f)


import matplotlib.pyplot as plt

with open("resultados_lambda_exp.json",'r') as f:
    result_paths=json.load(f)

# plot lambdas vs accuracy
lambdas = []
accs = []
val_acc_col = 1
for i in range(len(result_paths['exp_path'])):
    path_resx = result_paths['exp_path'][i]
    lambda_x = result_paths['lambda'][i]

    path_accs = os.path.join(path_resx, 'accuracy_simple.csv')
    data=pd.read_csv(path_accs,header=None)
    acc_last_it = data.iloc[-1][val_acc_col]

    lambdas.append(lambda_x)
    accs.append(acc_last_it)
print(sorted(zip(lambdas,accs),key=lambda x : x[0] ))
print(1.0/(14*14*64)*np.power(0.2,2))
accs.append(0.752)
lambdas.append(min(lambdas)*0.2)
accs.append(0.7362)
lambdas.append(min(lambdas)*0.2*0.2)

pairs=np.array(list(sorted(zip(lambdas,accs),key=lambda x : x[0]) ))


with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.semilogx(pairs[:,0],pairs[:,1],'*--')
    ax.grid()
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Accuracy validacion')
    plt.savefig('seleccion_lambda.png',dpi=100)