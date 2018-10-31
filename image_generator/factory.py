from image_generator.replace_with_crops import Replace_with_dataset_crops, \
    insert_crop_random_image
from image_generator.yu2018 import yu2018generative
import json

from select_tool.config_data import dataset_obj_dict


availables = ['yu2018','random_crop','random_insert']

def get_generator_from_key(key,dataset=None):


    if key == availables[0]:
        return yu2018generative()
    elif key == availables[1]:
        assert(dataset is not None),"Trying to use random_crop generator without dataset parameter"
        return Replace_with_dataset_crops(dataset,3)
    elif key == availables[2]:
        assert(dataset is not None),"Trying to use insert_crop_random_image generator without dataset parameter"
        return insert_crop_random_image(dataset,3)

    else:
        raise Exception("{0} not in available. Availables: {1}".format(key,availables))


    pass



if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser(description='Execute generative')
    parser.add_argument('select_config_json', help='The config_json to train')
    parser.add_argument('method_name', help='Generative algorithm to use. Available: {0}'.format(availables))

    args = parser.parse_args()
    config_path = args.select_config_json
    method_name = args.method_name

    with open(config_path,'r') as f:
        data_select = json.load(f)

    train_result_path = data_select['train_result_path']
    mask_file = data_select['mask_file']
    index_list = data_select['index_list']

    with open(train_result_path,'r') as f:
        data_t_r = json.load(f)
        path_train_file = data_t_r["train_file_used"]

        with open(path_train_file,'r') as f2:
            data_train_file = json.load(f2)

        d_k = data_train_file['dataset_key']
        d_p = data_train_file['dataset_params']

    d_obj = dataset_obj_dict[d_k](1, 1, **d_p)
    t= get_generator_from_key(method_name,dataset=d_obj)


    t.generate(d_obj,index_list,mask_file,config_path)
