
from subprocess import Popen, PIPE, STDOUT
import os 
import json
from exp_backprops import do_train_config
from contextlib import contextmanager
import time

@contextmanager
def show_elapsed(prefix=""):
    '''log the time usage in a code block
    prefix: the prefix text to show
    '''
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed_seconds = float("%.2f" % (end - start))
        print('{0} : elapsed seconds: {1}'.format( prefix, elapsed_seconds))

experimental_results_folder = 'test_exp'
os.makedirs(experimental_results_folder, exist_ok=True)


train_result_path = 'config_files/train_result/QuickDraw_Dataset__imagenet_classifier_cam_loss__19_Jan_2019__14_25.json'
"""
{
"train_mode" : "gen_train",
"model_load_path" : null,
"train_params" : {"gen_file" : "./gen_images/QuickDraw_Dataset_Replace_with_dataset_crops_20_Jan_2019__11_11/exp_details.json",
"just_eval" : false
 }
}

"""
"""
{
"train_mode" : "epochs",
    "dataset_key": "quickdraw",
    "model_key": "imagenet_cam_loss",
    "train_params" : {"epochs" : 1},
    "model_load_path" : null,

    "model_params": {},
    "dataset_params": {
        "data_folder": "./temp/quickdraw_expanded_images"
    }
}

"""

"""
{"train_file_used": "config_files/train_files/quickdraw_train_1.json",
 "mask_files": ["config_files/mask_files/mask_dummy_quickdraw_imagenet_cam_loss_2019-01-19__20:09:04.pkl"],
    "model_load_path": "model/imagenet_classifier/19_Jan_2019__14_25"}
"""

""" select file
{
    "index_list": [
        "5839801657851904",
        "5839805466279936"
    ],
    "train_result_path": "/home/aferral/PycharmProjects/generative_supervised_data_augmentation/config_files/train_result/QuickDraw_Dataset__imagenet_classifier_cam_loss__19_Jan_2019__14_25.json",
    "details": "",
    "mask_file": "config_files/mask_files/mask_dummy_quickdraw_imagenet_cam_loss_2019-01-19__20:09:04.pkl"
}
"""

""" GEN file
{
    "date": "2019-01-20 11:11:48",
    "dataset": "QuickDraw_Dataset",
    "used_select": "config_files/select_files/imagenet_cam_loss_quickdraw_20_Jan_2019__11_10_selection.json",
    "index_map": {
        "5839801657851904": [
            "gen_images/QuickDraw_Dataset_Replace_with_dataset_crops_20_Jan_2019__11_11/5839801657851904__0.png",
            "gen_images/QuickDraw_Dataset_Replace_with_dataset_crops_20_Jan_2019__11_11/5839801657851904__1.png",
            "gen_images/QuickDraw_Dataset_Replace_with_dataset_crops_20_Jan_2019__11_11/5839801657851904__2.png"
        ],
        "5839805466279936": [
            "gen_images/QuickDraw_Dataset_Replace_with_dataset_crops_20_Jan_2019__11_11/5839805466279936__0.png",
            "gen_images/QuickDraw_Dataset_Replace_with_dataset_crops_20_Jan_2019__11_11/5839805466279936__1.png",
            "gen_images/QuickDraw_Dataset_Replace_with_dataset_crops_20_Jan_2019__11_11/5839805466279936__2.png"
        ]
    },
    "mask_file": "config_files/mask_files/mask_dummy_quickdraw_imagenet_cam_loss_2019-01-19__20:09:04.pkl",
    "generator": "Replace_with_dataset_crops"
}
"""

with open(train_result_path,'r') as f:
    data_train=json.load(f)

    model_load_path = data_train['model_load_path']
    train_file_used = data_train['train_file_used']
    with open(train_file_used) as f2:
      data = json.load(f2)

      data_key = data['dataset_key']
      model_key = data['model_key']
      model_params = data['model_params']
      dataset_params = data['dataset_params']


# read train file used for dataset_key, model_key, store model_load_path, model_params, dataset_params
temp_train_eval_file = os.path.join(experimental_results_folder ,'temp_eval.json')

with open(temp_train_eval_file,'w') as f:
    dict_eval = {}
    dict_eval['train_mode'] = "epochs"
    dict_eval['dataset_key'] = data_key
    dict_eval['model_key'] = model_key
    dict_eval['train_params'] = {'epochs' : 0, 'just_eval' : True}
    dict_eval['model_load_path'] = model_load_path
    dict_eval['model_params'] = model_params
    dict_eval['dataset_params'] = dataset_params
    json.dump(dict_eval,f)

# generate select, gen files

temp_sel_file = os.path.join(experimental_results_folder ,'temp_sel.json')
with open(temp_sel_file,'w') as f:
    dict_eval = {}
    dict_eval['index_list'] = []
    dict_eval['train_result_path'] = train_result_path
    dict_eval['details'] = ''
    dict_eval['mask_file'] = None
    json.dump(dict_eval,f)


temp_train_genl_file = os.path.join(experimental_results_folder ,'temp_gen.json')
with open(temp_train_genl_file,'w') as f:
    dict_eval = {}
    dict_eval['date'] = ""
    dict_eval['dataset'] = ''
    dict_eval['used_select'] = temp_sel_file
    dict_eval['index_map'] = {}
    dict_eval['mask_file'] = None
    dict_eval['model_params'] = model_params
    dict_eval['generator'] = ''
    json.dump(dict_eval,f)

temp_train_file_gen = os.path.join(experimental_results_folder ,'temp_train_for_GEN.json')
with open(temp_train_file_gen,'w') as f:
    dict_eval = {}
    dict_eval['train_mode'] = "gen_train"
    dict_eval['model_load_path'] = None
    dict_eval['train_params'] = {"gen_file" : temp_train_genl_file,"just_eval" : False}

    json.dump(dict_eval,f)




# run eval get output to eval result
# print('DOING EVAL')
# with show_elapsed(prefix="Eval time"):
#     cmd = ["python", "t.py",temp_train_eval_file]
#     out_eval_txt = os.path.join(experimental_results_folder,"result_eval.txt")
#     with open(out_eval_txt,'w') as f:
#         p = Popen(cmd, stdin=None, stdout=f,universal_newlines=True)
#         p.wait()



# # run to get continued training exp
# print('DOING CONTINUED TRAINING')
# with show_elapsed(prefix="continuated train time"):
#     out_eval_txt = os.path.join(experimental_results_folder,"result_cont_training.txt")
#     with open(out_eval_txt,'w') as f:
#         p = Popen(["python", "t.py",temp_train_file_gen], stdin=PIPE, stdout=f,universal_newlines=True)
#         stdou,stderr = p.communicate(input="13\n1\n6\n")



# generate missclass analisis
#print("DOING MISSCLASS ANALISIS")
#with show_elapsed(prefix="continuated train time"):
#    exp_params_dict = {}
#    exp_params_dict['indexs_string'] = ''
#    exp_params_dict['n_iterations'] = 0  # 15,3, 50, 25
#    exp_params_dict['iter_till_insert'] = 0
#    exp_params_dict['gens_per_original'] = 0
#    exp_params_dict['skip_insert'] = False
#    exp_params_dict['batch_size_exp'] = 0
#    exp_params_dict['op'] = ''
#    exp_params_dict['sel_dropout'] = False
#    exp_params_dict['exp_list'] = [5]
#    exp_params_dict['base_name'] = 'miss_class_analisis'
#    exp_params_dict['dropout_k']= 16
#    exp_params_dict['mask_file_path_map'] = None  # 'mask_map_label_9__2018-Dec-12--00:04.pkl'
#    exp_params_dict['plot_masks'] = False
#
#    do_train_config(temp_train_file_gen,out_file_path=experimental_results_folder, **exp_params_dict )

# generate cluster visualization
print('DOING CLUSTER CAM ')
with show_elapsed(prefix="Vis cluster"):
    out_folder_cam_vis = os.path.join(experimental_results_folder,'cam_vis_exp')
    os.makedirs(out_folder_cam_vis,exist_ok=True)
    acts_pkl = os.path.join(out_folder_cam_vis,'acts.pkl')
    acts_reduced = os.path.join(out_folder_cam_vis,'acts_dim_red.pkl')
    cmd = ["python", "-m",'vis_exp.cluster_cam_acts',temp_train_file_gen,'vgg_16/conv5/conv5_3/Relu:0','--use_pred_max','--activations_file_path',acts_pkl,'--out_path_dim_red',acts_reduced]
    p = Popen(cmd)
    p.wait()
  
# TODO CLUSTER 2
#/home/aferral/PycharmProjects/generative_supervised_data_augmentation/vis_exp/cluster_cam_acts_2.py


# TODO wait for mask selection



  # usar perdida mostrando solo imagenes selecionadas
  # usar perdida con focal loss
  # usar perdida con CAM
# TODO here
print("DOING CAM loss ANALISIS")
with show_elapsed(prefix="train cam loss"):
   exp_params_dict = {}
   exp_params_dict['indexs_string'] = ''
   exp_params_dict['n_iterations'] = 30  # 15,3, 50, 25
   exp_params_dict['iter_till_insert'] = 1
   exp_params_dict['gens_per_original'] = 0
   exp_params_dict['skip_insert'] = False
   exp_params_dict['batch_size_exp'] = 50
   exp_params_dict['sel_dropout'] = False
   exp_params_dict['exp_list'] = [3]
   exp_params_dict['base_name'] = 'miss_class_analisis'
   exp_params_dict['dropout_k']= 16
   exp_params_dict['mask_file_path_map'] = None  # 'mask_map_label_9__2018-Dec-12--00:04.pkl'
   exp_params_dict['plot_masks'] = False
   do_train_config(temp_train_file_gen,out_file_path=experimental_results_folder, **exp_params_dict )

# TODO here
print("DOING just original")
with show_elapsed(prefix="train cam loss"):
   exp_params_dict = {}
   exp_params_dict['indexs_string'] = ''
   exp_params_dict['n_iterations'] = 0  # 15,3, 50, 25
   exp_params_dict['iter_till_insert'] = 0
   exp_params_dict['gens_per_original'] = 0
   exp_params_dict['skip_insert'] = False
   exp_params_dict['batch_size_exp'] = 0
   exp_params_dict['op'] = ''
   exp_params_dict['sel_dropout'] = False
   exp_params_dict['exp_list'] = [0]
   exp_params_dict['base_name'] = 'miss_class_analisis'
   exp_params_dict['dropout_k']= 16
   exp_params_dict['mask_file_path_map'] = None  # 'mask_map_label_9__2018-Dec-12--00:04.pkl'
   exp_params_dict['plot_masks'] = False
   do_train_config(temp_train_file_gen,out_file_path=experimental_results_folder, **exp_params_dict )
