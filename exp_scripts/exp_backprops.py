import types
from contextlib import ExitStack
import random
import cv2
import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import time
# From http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
from classification_models.classification_model import CWR_classifier
from classification_models.imagenet_subset_cam_loss import \
    imagenet_classifier_cam_loss
from classification_models.imagenet_subset_cam_loss_auto import \
    imagenet_classifier_cam_loss_AUTO
from classification_models.imagenet_subset_cam_loss_v2 import \
    imagenet_classifier_cam_loss_V2
from classification_models.imagenet_subset_conv_loss import \
    imagenet_classifier_CONV_LOSS
from classification_models.imagenet_subset_focal_loss import \
    imagenet_classifier_focal_loss
from classification_models.simple_model import simple_classifier
from classification_models.vgg16_edited import vgg_16_CAM
from classification_models.vgg_16_batch_norm import vgg_16_batchnorm
from datasets.cifar10_data import Cifar10_Dataset
from datasets.cwr_dataset import CWR_Dataset
from datasets.dataset import Dataset, Digits_Dataset, placeholder_dataset
from image_generator.factory import get_generator_from_key
from image_generator.yu2018 import yu2018generative
from select_tool.config_data import model_obj_dict, dataset_obj_dict
from select_tool.img_selector import call_one_use_select
from utils import show_graph, now_string, timeit, load_mask, imshow_util, \
    get_img_cam_index, get_img_RAW_cam_index, parse_config_recur
import json
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from t import dropout_mask, calc_dropout_mask, get_feed_dict_fun, \
    flush_to_dataset, generate_random_for_loop, get_img_cam, do_backprop, \
    sel_mask, eval_current_model, generate_random_from_list



def auto_cam_loss(model, sample_fun,
    data_container,
    base_data,
    indexs_to_plot,
    out_f,
    gen_mask_map,
    config_path,
    optional_backprop_args, indexs_to_improve):


    def feed_dict_to_improve(original_fun,b, inds_to_improve,use_simple_index=False, k = None):
        def get_fd(self, is_train=True, debug=True):

            base_fd = original_fun(is_train=is_train, debug=True)
            ex_input = base_fd['model_input:0']

            conv_acts, softmax_w, indexs = self.sess.run(
                [self.last_conv, self.softmax_weights, 'indexs_input:0'],
                base_fd)

            batch_size = ex_input.shape[0]

            base_cam_mask = np.zeros((batch_size,))

            for i in range(len(indexs)):
                ind_img = indexs[i].decode('utf-8') if not use_simple_index else \
                indexs[i].decode('utf-8').split('__')[1]
                if ind_img in inds_to_improve:
                    base_cam_mask[i] = 1

            base_fd['cam_loss_term/use_cam_loss:0'] = base_cam_mask

            return base_fd

        return get_fd


    def check_all_correct(dataset,indexs_to_improve):
        indexs_still_to_improve = []

        for ind in indexs_to_improve:
            im,lb = dataset.get_train_image_at(ind)
            image, conv_acts, softmax_w, pred = model.feed_forward_vis(im)

            if pred.argmax() != lb[0]:
                indexs_still_to_improve.append(ind)

        return indexs_still_to_improve



    it = 0
    n_bp = 0
    inds_to_fix = indexs_to_improve
    while it < 100:

        inds_to_fix =check_all_correct(base_data, inds_to_fix)
        feed_fun = lambda a, b, k=None: feed_dict_to_improve(a, b, inds_to_fix,
                                                        use_simple_index=False,
                                                        k=None)
        print("Len to fix {0}".format(inds_to_fix))
        if len(inds_to_fix) == 0:
            break

        # generate random batch
        index_list, img_list, label_list, _ = sample_fun(base_data)

        # add selected images
        for img_ind in indexs_to_improve:
            img, label, all_cams, scores, r_label = get_img_cam(img_ind, base_data, model,show_images=False)
            index_list.append(img_ind)
            img_list.append(img.squeeze())
            label_list.append(label)

        # flush
        flush_to_dataset(data_container, index_list,img_list, label_list, False,None,None,None)

        # do backpropagation
        if 'new_feed_fun' in optional_backprop_args:
            optional_backprop_args.pop('new_feed_fun')
        n_bp = do_backprop(model, base_data, data_container,
                                indexs_to_plot, [],
                                out_f, gen_mask_map,
                           n_bp, config_path,new_feed_fun=feed_fun,**optional_backprop_args)
        it+=1




def for_loop_model_exp(model, sample_fun,
    data_container,
    base_data,
    i_untill_inser,
    skp_insert,
    indexs_to_plot,
    gen_per_org,
    out_f,
    gen_mask_map,
    config_path,
    optional_backprop_args,n_iterations,op):

    backprops=0

    for i in range(n_iterations):

        # generate random batch
        index_list, img_list, label_list, _ = sample_fun(base_data)


        if i % i_untill_inser == 0 and not(skp_insert):
            # add original images
            for img_ind in indexs_to_plot:
                img, label, all_cams, scores, r_label = get_img_cam(img_ind, base_data, model,show_images=False)
                index_list.append(img_ind)
                img_list.append(img.squeeze())
                label_list.append(label)

            # add generated images if any
            if op == 'add_gen':
                # generate random batch
                for img_ind in gen_mask_map:
                    sel_cam, current_mask = gen_mask_map[img_ind]
                    img, label, all_cams, scores, r_label = get_img_cam(img_ind, base_data, model,show_images=False)
                    i_list, im_list, l_list, c_gens = generate_random_for_loop(base_data, img_ind, img.squeeze(),label, 0,current_mask, n_random=0,gen_imgs=gen_per_org)

                    index_list += i_list
                    img_list += im_list
                    label_list += l_list

        # flush
        flush_to_dataset(data_container, index_list,img_list, label_list, False,None,None,None)

        # do backpropagation
        backprops = do_backprop(model, base_data, data_container,
                                indexs_to_plot, [],
                                out_f, gen_mask_map,
                                backprops, config_path,**optional_backprop_args)

    return

def get_f_d_cam_loss(original_fun, gen_mask_map,lambda_param,k=None,use_simple_index=False):
    def manual_dropout(self, is_train=True, debug=True):

        base_fd = original_fun(is_train=is_train, debug=True)
        ex_input = base_fd['model_input:0']

        conv_acts, softmax_w, indexs = self.sess.run(
            [self.last_conv, self.softmax_weights, 'indexs_input:0'], base_fd)

        batch_size = ex_input.shape[0]
        h, w, c = conv_acts.shape[1], conv_acts.shape[2], conv_acts.shape[3]

        base_cam_mask = np.zeros((batch_size, h, w))
        base_fd['cam_loss_term/loss_lambda:0'] = lambda_param
        print("Using lambda {0}".format(lambda_param))

        def clean_index(raw_ind):
            str_ind = raw_ind.decode('utf-8')
            if 'gen_id__' in str_ind:
                return str_ind.split('__')[1]
            else:
                return str_ind

        for i in range(len(indexs)):
            ind_img = clean_index(indexs[i])
            if ind_img in gen_mask_map:
                sel_cam, current_mask = gen_mask_map[ind_img]
                down_sampled = cv2.resize(current_mask.astype(np.float32),(h, w))
                base_cam_mask[i] = down_sampled

        base_fd['cam_loss_term/cam_mask:0'] = base_cam_mask

        return base_fd

    return manual_dropout

def calc_missclass_and_plot(model_class, model_params,model_load_path,dataset,out_folder,indexs_list=None):
    print("Doing missclass analisis")

    with model_class(dataset,**model_params) as model:

        if not(model_load_path is None):
            model.load(model_load_path)

        model.dataset.initialize_iterator_train(model.sess)
        counter = 0

        # get  index of missclasified
        if indexs_list is None:
            inds_mis_class = []
            pred_labels = []
            r_labels = []
            while True:
                try:
                    fd = model.prepare_feed(is_train=False)

                    # extract activations in batch, CAM
                    index_batch, soft_max_out, targets = model.sess.run([model.indexs, model.pred,model.targets], feed_dict=fd)

                    # calc missclass_indexs

                    predicted = soft_max_out.argmax(axis=1)
                    selection = (predicted != targets.argmax(axis=1))

                    pred_labels += predicted.tolist()
                    r_labels += targets.argmax(axis=1).tolist()
                    inds_mis_class += index_batch[selection].tolist()


                    if counter % 50 == 0:
                        print(counter)
                    counter += 1

                except tf.errors.OutOfRangeError:
                    break

            inds_mis_class = list(map(lambda x : x.decode('utf8') if type(x) == bytes else x,inds_mis_class))
            inds_mis_class = list(set(inds_mis_class))

            # get confusion matrix on dataset
            conf_m=confusion_matrix(r_labels, pred_labels)


            with open(os.path.join(out_folder,"conf_matrix.txt"),'w') as f:
                f.write('Accuracy {0} \n'.format(accuracy_score(r_labels,pred_labels)))
                f.write(str(conf_m))
        else:
            inds_mis_class = indexs_list


        # for index get cam (image-eval), image, preds
        print("About to plot {0} indexs".format(len(inds_mis_class)))
        if len(inds_mis_class) > 1500:
            random.seed(3)
            print("Capping at 1500 indexs ".format(len(inds_mis_class)))
            inds_mis_class = random.sample(inds_mis_class,1500)

        eval_current_model('eval', model, dataset, inds_mis_class, '', out_folder, None, eval=False,save_img=True)

    return inds_mis_class




def do_train_config(config_path,
                    indexs_string,
                    n_iterations,
                    iter_till_insert,
                    gens_per_original,
                    skip_insert,
                    batch_size_exp,
                    exp_list,
                    base_name,
                    dropout_k,
                    mask_file_path_map,
                    plot_masks,
                    out_file_path=None,
                    missclass_index_path=None, add_summary=False,
                    lambda_value =  0.2*0.2*1.0/(14*14) ,
                    param_a=None,
                    param_b=None,
                    tf_config=None
                   ):

    """
        # #'n02114548_10505.JPEG'
        #"n02120079_1801.JPEG,n02114548_6413.JPEG,n02120079_4625.JPEG,n02114548_3599.JPEG,n02120079_6223.JPEG,n02114548_10182.JPEG,n02120079_13994.JPEG,n02120079_12464.JPEG,n02120079_14956.JPEG"
        indexs_string = 'n02114548_10505.JPEG,n02120079_9808.JPEG,n02114548_11513.JPEG,n02120079_4409.JPEG,n02114548_5207.JPEG'
        n_iterations = 50 # 15,3, 50, 25
        iter_till_insert = 3
        gens_per_original = 1
        skip_insert = False
        batch_size_exp = 50
        op = ''
        sel_dropout = False
        exp_list = [3]
        base_name = 'exp_cam_loss'
        dropout_k = 16
        mask_file_path_map = "sel_mask_exp_3.pkl" #'mask_map_label_9__2018-Dec-12--00:04.pkl'
        plot_masks = True


    """
    sel_dropout = False

    data_config = parse_config_recur(config_path)
    t_params = data_config['train_params']
    m_k = data_config['model_key']
    m_p = data_config['model_params']
    d_k = data_config['dataset_key']
    d_p = data_config['dataset_params']

    m_p['use_summary']= add_summary
    if tf_config:
        m_p['tf_config'] = tf_config

    if data_config['model_load_path_at_train_file'] is not None:
        model_load_path = data_config['model_load_path_at_train_file']
        print("Using model load path from TRAIN_FILE {0}".format(model_load_path))
    else:
        model_load_path = data_config['model_load_path_train_result']
        print("Using model load path from SELECT_FILE {0}".format(model_load_path))


    batch_size = t_params['b_size'] if 'b_size' in t_params else 20
    epochs = t_params['epochs'] if 'epochs' in t_params else 1
    model_class = model_obj_dict[m_k]
    dataset_class = dataset_obj_dict[d_k]



    gen_mask_map = {}

    if not(mask_file_path_map is None):
        with open(mask_file_path_map, 'rb') as f:
            gen_mask_map = pickle.load(f)
            if 'masks' in gen_mask_map:
                gen_mask_map = {k: (None, gen_mask_map['masks'][k]) for k in gen_mask_map['masks']}


    special_batch_count = 0



    for id_exp in exp_list:

        plt.close('all')

        optional_backprop_args = {}

        sample_batch_fun  = lambda x : generate_random_for_loop(x, None, None, None, 0, None, n_random=batch_size_exp,gen_imgs=0)

        if id_exp == 0:
            exp_type = 'add_org'
            op = ''
            sel_dropout = False

        elif id_exp == 1:
            exp_type = 'gen'
            op = 'add_gen'
            sel_dropout = False

        elif id_exp == 2:
            exp_type = 'dropout_org'
            op = ''
            sel_dropout = True

        elif id_exp == 3:
            model_class = imagenet_classifier_cam_loss
            exp_type = 'cam_loss'
            op = ''
            sel_dropout = True

            optional_backprop_args['new_feed_fun'] = lambda f, gm,k=None: get_f_d_cam_loss(f, gm, lambda_value, k=k,use_simple_index=False)

        elif id_exp == 12:
            model_class = imagenet_classifier_CONV_LOSS
            exp_type = 'CONV_loss'
            op = ''
            sel_dropout = True

            optional_backprop_args['new_feed_fun'] = lambda f, gm,k=None: get_f_d_cam_loss(f, gm, lambda_value, k=k,use_simple_index=False)


        elif id_exp == 9:
            model_class = imagenet_classifier_cam_loss_V2
            exp_type = 'cam_V2_loss'
            op = ''
            sel_dropout = True
            m_p['param_b'] = param_b
            m_p['param_a'] = param_a

            optional_backprop_args['new_feed_fun'] = lambda f, gm,k=None: get_f_d_cam_loss(f, gm, lambda_value, k=k,use_simple_index=False)

        elif id_exp == 20:
            exp_type = 'cam_V2__KEEP_MODEL_loss'
            op = ''
            sel_dropout = True

            optional_backprop_args['new_feed_fun'] = lambda f, gm,k=None: get_f_d_cam_loss(f, gm, lambda_value, k=k,use_simple_index=False)



        elif id_exp == 4:
            model_class = imagenet_classifier_focal_loss
            exp_type = 'focal_loss'
            op = ''
            sel_dropout = False

        elif id_exp == 5:
            exp_type = 'just_missclass'

        elif id_exp == 6:
            exp_type = 'cont_training'
            op = ''
            skip_insert = True
            sel_dropout = False

        elif id_exp == 7:
            exp_type = 'cont_training_big_sample_missclass'
            op = ''
            skip_insert = True
            sel_dropout = False
            assert(missclass_index_path is not None),'Give missclass_index_list parameter for exp'

            with open(missclass_index_path,'r') as f:
                missclass_index_list = f.read().split('\n')
            missclass_index_list = list(filter(lambda x : len(x) > 0 , missclass_index_list))
            n_missclass = int(batch_size_exp*0.3)
            n_random = int(batch_size_exp*0.7)
            sample_batch_fun = lambda x: generate_random_from_list(x, missclass_index_list, 0,n_random=n_random, gen_imgs=n_missclass)

            pass
        
        elif id_exp == 8:
            exp_type = 'training_all_masks'
            op = ''
            model_class = imagenet_classifier_cam_loss
            skip_insert = True
            sel_dropout = True
            assert(missclass_index_path is not None),'Give missclass_index_list parameter for exp'
            assert(mask_file_path_map is not None),'GIve mask file'

            with open(missclass_index_path,'r') as f:
                missclass_index_list = f.read().split('\n')
            missclass_index_list = list(filter(lambda x : len(x) > 0 , missclass_index_list))

            n_with_mask = int(batch_size_exp*0.7)
            n_random = int(batch_size_exp*0.3)
            sample_batch_fun = lambda x: generate_random_from_list(x, missclass_index_list, 0,n_random=n_random, gen_imgs=n_with_mask)

            optional_backprop_args['new_feed_fun'] = lambda f, gm, k=None: get_f_d_cam_loss( f, gm, lambda_value, k=k,use_simple_index=True)

            with open(mask_file_path_map, 'rb') as f:
                t = pickle.load(f)['masks']
                gen_mask_map = {k : (None,t[k]) for k in t}

        elif id_exp == 10:
            exp_type = 'training_all_masks_V2'
            op = ''
            model_class =  imagenet_classifier_cam_loss_V2
            sel_dropout = True
            assert(missclass_index_path is not None),'Give missclass_index_list parameter for exp'
            assert(mask_file_path_map is not None),'GIve mask file'

            with open(missclass_index_path,'r') as f:
                missclass_index_list = f.read().split('\n')
            missclass_index_list = list(filter(lambda x : len(x) > 0 , missclass_index_list))

            n_with_mask = int(batch_size_exp*0.7)
            n_random = int(batch_size_exp*0.3)
            sample_batch_fun = lambda x: generate_random_from_list(x, missclass_index_list, 0,n_random=n_random, gen_imgs=n_with_mask)

            optional_backprop_args['new_feed_fun'] = lambda f, gm, k=None: get_f_d_cam_loss( f, gm, lambda_value, k=k,use_simple_index=False)

            with open(mask_file_path_map, 'rb') as f:
                t = pickle.load(f)['masks']
                gen_mask_map = {k : (None,t[k]) for k in t}

        elif id_exp == 31:
            exp_type = 'training_all_masks_V2'
            op = ''
            model_class =  simple_classifier
            sel_dropout = True
            assert(missclass_index_path is not None),'Give missclass_index_list parameter for exp'
            assert(mask_file_path_map is not None),'GIve mask file'

            with open(missclass_index_path,'r') as f:
                missclass_index_list = f.read().split('\n')
            missclass_index_list = list(filter(lambda x : len(x) > 0 , missclass_index_list))

            n_with_mask = int(batch_size_exp*0.7)
            n_random = int(batch_size_exp*0.3)
            sample_batch_fun = lambda x: generate_random_from_list(x, missclass_index_list, 0,n_random=n_random, gen_imgs=n_with_mask)

            optional_backprop_args['new_feed_fun'] = lambda f, gm, k=None: get_f_d_cam_loss( f, gm, lambda_value, k=k,use_simple_index=False)

            with open(mask_file_path_map, 'rb') as f:
                t = pickle.load(f)['masks']
                gen_mask_map = {k : (None,t[k]) for k in t}



        elif id_exp == 30:
            exp_type = 'training_masks_features_espurias_V2'
            op = ''
            model_class =  imagenet_classifier_cam_loss_V2
            sel_dropout = True
            assert(mask_file_path_map is not None),'GIve mask file'

            with open(mask_file_path_map, 'rb') as f:
                gen_mask_map = pickle.load(f)

            missclass_index_list = list(gen_mask_map.keys())

            n_with_mask = int(batch_size_exp*0.7)
            n_random = int(batch_size_exp*0.3)
            sample_batch_fun = lambda x: generate_random_from_list(x, missclass_index_list, 0,n_random=n_random, gen_imgs=n_with_mask)

            optional_backprop_args['new_feed_fun'] = lambda f, gm, k=None: get_f_d_cam_loss( f, gm, lambda_value, k=k,use_simple_index=False)





        elif id_exp == 13:
            exp_type = 'training_all_masks_CONV_LOSS'
            op = ''
            model_class = imagenet_classifier_CONV_LOSS #simple_classifier
            skip_insert = True
            sel_dropout = True
            assert(missclass_index_path is not None),'Give missclass_index_list parameter for exp'
            assert(mask_file_path_map is not None),'GIve mask file'

            with open(missclass_index_path,'r') as f:
                missclass_index_list = f.read().split('\n')
            missclass_index_list = list(filter(lambda x : len(x) > 0 , missclass_index_list))

            n_with_mask = int(batch_size_exp*0.7)
            n_random = int(batch_size_exp*0.3)
            sample_batch_fun = lambda x: generate_random_from_list(x, missclass_index_list, 0,n_random=n_random, gen_imgs=n_with_mask)

            optional_backprop_args['new_feed_fun'] = lambda f, gm, k=None: get_f_d_cam_loss( f, gm, lambda_value, k=k,use_simple_index=True)

            with open(mask_file_path_map, 'rb') as f:
                t = pickle.load(f)['masks']
                gen_mask_map = {k : (None,t[k]) for k in t}



        elif id_exp == 11:
            exp_type = 'auto_training_CAM'
            op = ''
            model_class = imagenet_classifier_cam_loss_AUTO
            sel_dropout = True

        optional_backprop_args['change_feed_dict'] = sel_dropout
        optional_backprop_args['selective_dropout_k'] = dropout_k


        exp_name = '{0}_{1}_{2}'.format(base_name,exp_type,now_string())
        out_f = out_file_path if not(out_file_path is None) else os.path.join('out_backprops', exp_name)
        m_p['out_folder'] = out_f

        selected_img_indexs = list(
            map(lambda x: x.strip(), indexs_string.split(",")))

        do_exp_fun = lambda md,d_dummy, d_base :  for_loop_model_exp(md, sample_batch_fun,d_dummy,d_base,
                                                                     iter_till_insert,skip_insert,selected_img_indexs,
                                                                     gens_per_original,out_f,gen_mask_map,config_path,
                                                                     optional_backprop_args, n_iterations, op)


        if id_exp == 11:
            do_exp_fun = lambda md, d_dummy, d_base: auto_cam_loss(md, sample_batch_fun,
                          d_dummy,
                          d_base,selected_img_indexs,
                          out_f,
                          gen_mask_map,
                          config_path,
                          optional_backprop_args, selected_img_indexs)




        with tf.Graph().as_default():
            base_dataset = dataset_class(epochs, batch_size,**d_p)  # type: Dataset

            st_missclass_folder = os.path.join(out_f, 'start')
            os.makedirs(st_missclass_folder, exist_ok=True)
            st_missclass_list = calc_missclass_and_plot(model_class, m_p,model_load_path, base_dataset,st_missclass_folder)

        if id_exp == 5:
            return


        with tf.Graph().as_default():


            base_dataset = dataset_class(epochs, batch_size, **d_p)  # type: Dataset
            # Create dummy dataset add all gen_images and random images
            dataset_one_use = placeholder_dataset(base_dataset)

            if plot_masks:
                out_folder_originals = os.path.join(out_f, 'originals')
                os.makedirs(out_folder_originals, exist_ok=True)
                for k in gen_mask_map:
                    org_img, label = base_dataset.get_train_image_at(k)

                    out_name = os.path.join(out_folder_originals,'{0}_mask.png'.format(k))
                    out_name_or = os.path.join(out_folder_originals, '{0}.png'.format(k))

                    rgb = (len(org_img[0].shape) == 3 and org_img[0].shape[2] == 3)

                    img_out_or = cv2.cvtColor(org_img[0],cv2.COLOR_RGB2BGR) if rgb else org_img[0]


                    img_out = gen_mask_map[k][1].astype(np.uint8)*255
                    cv2.imwrite(out_name,img_out)
                    cv2.imwrite(out_name_or, img_out_or)

                    #
                    # img_out = gen_mask_map[k][1].astype(np.float32)
                    # plt.imshow(img_out)
                    # plt.savefig(out_name)
                    # plt.imshow(img_out_or)
                    # plt.savefig(out_name_or)
                    # plt.close('all')


            with model_class(dataset_one_use, **m_p) as model: # this also save the train_result
                model.load(model_load_path)


                do_exp_fun(model,dataset_one_use,base_dataset)


                saver = tf.train.Saver()
                folder_model_temp = 'temp_folder_model'
                os.makedirs(folder_model_temp,exist_ok=True)
                out_model=model.save_model(saver,folder_model_temp)

        with tf.Graph().as_default():
            base_dataset = dataset_class(epochs, batch_size, **d_p)  # type: Dataset

            en_missclass_folder = os.path.join(out_f,'end')
            os.makedirs(en_missclass_folder, exist_ok=True)
            st_miss_class = calc_missclass_and_plot(model_class, m_p,out_model,base_dataset, en_missclass_folder)

        with tf.Graph().as_default():
            base_dataset = dataset_class(epochs, batch_size,
                                         **d_p)  # type: Dataset

            after_missclass_folder = os.path.join(out_f,'start_indexs_after')
            os.makedirs(after_missclass_folder, exist_ok=True)
            st_miss_class = calc_missclass_and_plot(model_class, m_p,out_model,base_dataset, after_missclass_folder,indexs_list=st_missclass_list)

        out_model_path = os.path.join(out_f, 'model_after')

        import shutil
        shutil.move(out_model,out_model_path)

    return out_f





if __name__ == '__main__':




    """
    Indices imagenet_subset
    
    #'./config_files/train_files/gen_imagenet_subset_09_oct_074.json'
    
    'n02114548_10505.JPEG'
    "n02120079_1801.JPEG,n02114548_6413.JPEG,n02120079_4625.JPEG,n02114548_3599.JPEG,n02120079_6223.JPEG,n02114548_10182.JPEG,n02120079_13994.JPEG,n02120079_12464.JPEG,n02120079_14956.JPEG"
    
    
    "sel_mask_exp_3.pkl"  #'./config_files/mask_files/all_filtered_masks.pkl'   # 'mask_map_label_9__2018-Dec-12--00:04.pkl' #
    
    'all_masks_indexs.txt'#'./out_backprops/lista_indices.txt'
    
    './config_files/train_files/gen_imagenet_subset_09_oct_074.json'
    
    'all_masks_indexs.txt'#'./out_backprops/lista_indices.txt'
    
    # data aug
    ./config_files/train_result/Imagenet_Dataset__imagenet_classifier__20_Feb_2019__18_17.json
    
    config_file = './config_files/train_files/gen_imagenet_subset_09_oct_074.json'
    exp_params_dict = {}
    exp_params_dict['indexs_string'] = 'n02114548_10505.JPEG,n02120079_9808.JPEG,n02114548_11513.JPEG,n02120079_4409.JPEG,n02114548_5207.JPEG'
    exp_params_dict['n_iterations'] = 100  # 15,3, 50, 25
    exp_params_dict['iter_till_insert'] = 1
    exp_params_dict['gens_per_original'] = 1
    exp_params_dict['skip_insert'] = False
    exp_params_dict['batch_size_exp'] = 50
    exp_params_dict['exp_list'] = [10]
    exp_params_dict['base_name'] = 'loss_v2_ALL_MASKS_high_params'
    exp_params_dict['dropout_k']= 16
    exp_params_dict['mask_file_path_map'] = './config_files/mask_files/all_filtered_masks.pkl'
    exp_params_dict['plot_masks'] = True
    exp_params_dict['missclass_index_path'] = 'all_masks_indexs.txt'
    
    # Experiment con mascaras de features espurias 
    config_file = './config_files/train_files/gen_imagenet_subset_09_oct_074.json'
    exp_params_dict = {}
    exp_params_dict['indexs_string'] = 'n02114548_10505.JPEG,n02120079_9808.JPEG,n02114548_11513.JPEG,n02120079_4409.JPEG,n02114548_5207.JPEG'
    exp_params_dict['n_iterations'] = 100  # 15,3, 50, 25
    exp_params_dict['iter_till_insert'] = 1
    exp_params_dict['gens_per_original'] = 1
    exp_params_dict['skip_insert'] = False
    exp_params_dict['batch_size_exp'] = 50
    exp_params_dict['exp_list'] = [30]
    exp_params_dict['base_name'] = 'loss_v2_ALL_MASKS_ONLY_ONE_ESPURIAS_LAST_EXP'
    exp_params_dict['dropout_k']= 16
    exp_params_dict['mask_file_path_map'] = './exp_filters_espurios/mask_espurios_visuales.pkl' #./mask_espurios_visuales_centrada_39.pkl
    exp_params_dict['plot_masks'] = True
    exp_params_dict['missclass_index_path'] = 'all_masks_indexs.txt'
    
    
    
    
    Indices quickdraw
    
    '4503637852160000,4503693233750016,4503756869730304,4503790474493952,4503964252897280,4503976085028864,4504252619685888'
    
    ./config_files/mask_files/mask_dummy_quickdraw_quickdraw_2019-02-09__12:38:35.pkl
    
    './config_files/train_result/QuickDraw_Dataset__quickdraw_classifier__09_Feb_2019__12_23.json'
    './config_files/train_result/QuickDraw_Dataset__imagenet_classifier_cam_loss__19_Jan_2019__14_25.json'
    
    Mascaras version 2
    '4503637852160000,4905290711433216,6395809920712704,6613196754386944,4932170596483072,4644317375234048'
    'masks_v2_sketchs.pkl'
    
    config_file = './config_files/train_result/QuickDraw_Dataset__quickdraw_classifier__09_Feb_2019__12_23.json'
    exp_params_dict = {}
    exp_params_dict['indexs_string'] = '4503637852160000,4503693233750016,4503756869730304,4503790474493952,4503964252897280,4503976085028864,4504252619685888'
    exp_params_dict['n_iterations'] = 100  # 15,3, 50, 25
    exp_params_dict['iter_till_insert'] = 1
    exp_params_dict['gens_per_original'] = 1
    exp_params_dict['skip_insert'] = False
    exp_params_dict['batch_size_exp'] = 50
    exp_params_dict['exp_list'] = [10]
    exp_params_dict['base_name'] = 'sketchs_cont_training'
    exp_params_dict['dropout_k']= 16
    exp_params_dict['mask_file_path_map'] = './config_files/mask_files/mask_dummy_quickdraw_quickdraw_2019-02-09__12:38:35.pkl'
    exp_params_dict['plot_masks'] = True
    exp_params_dict['missclass_index_path'] = 'all_masks_indexs.txt'
    
    
    
    
    
    # exp enfocado en uuna caracteristica
    config_file = './config_files/train_files/gen_imagenet_subset_09_oct_074.json'
    exp_params_dict = {}
    exp_params_dict['indexs_string'] = 'n02114548_10505.JPEG,n02120079_9808.JPEG,n02114548_11513.JPEG,n02120079_4409.JPEG,n02114548_5207.JPEG'
    exp_params_dict['n_iterations'] = 100  # 15,3, 50, 25
    exp_params_dict['iter_till_insert'] = 1
    exp_params_dict['gens_per_original'] = 1
    exp_params_dict['skip_insert'] = False
    exp_params_dict['batch_size_exp'] = 50
    exp_params_dict['exp_list'] = [30]
    exp_params_dict['base_name'] = 'loss_v2_ALL_MASKS_multiples_espurias'
    exp_params_dict['dropout_k']= 16
    exp_params_dict['mask_file_path_map'] = './exp_filters_espurios/mask_espurios_visuales.pkl'
    exp_params_dict['plot_masks'] = True
    exp_params_dict['missclass_index_path'] = 'all_masks_indexs.txt'

    exp_params_dict['add_summary'] = True
    
    
 
    
    """

    config_file = './config_files/train_files/gen_imagenet_subset_09_oct_074.json'
    exp_params_dict = {}
    exp_params_dict['indexs_string'] = 'n02114548_10505.JPEG,n02120079_9808.JPEG,n02114548_11513.JPEG,n02120079_4409.JPEG,n02114548_5207.JPEG'
    exp_params_dict['n_iterations'] = 10  # 15,3, 50, 25
    exp_params_dict['iter_till_insert'] = 1
    exp_params_dict['gens_per_original'] = 1
    exp_params_dict['skip_insert'] = False
    exp_params_dict['batch_size_exp'] = 50
    exp_params_dict['exp_list'] = [30]
    exp_params_dict['base_name'] = 'loss_v2_ALL_MASKS_ONLY_ONE_ESPURIAS_LAST_EXP'
    exp_params_dict['dropout_k']= 16
    exp_params_dict['mask_file_path_map'] = './exp_filters_espurios/mask_espurios_visuales.pkl' #./mask_espurios_visuales_centrada_39.pkl
    exp_params_dict['plot_masks'] = True
    exp_params_dict['missclass_index_path'] = 'all_masks_indexs.txt'
    exp_params_dict['add_summary'] = True



    do_train_config(config_file,**exp_params_dict )
