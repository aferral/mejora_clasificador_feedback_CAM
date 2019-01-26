import cv2
import tensorflow as tf
import numpy as np
from skimage import measure
from sklearn.cluster import MiniBatchKMeans
from classification_models.imagenet_subset_cam_loss import \
    imagenet_classifier_cam_loss
from select_tool.config_data import model_obj_dict, dataset_obj_dict
from utils import parse_config_recur
from skimage.filters import threshold_otsu, rank
from scipy.sparse.csgraph import connected_components
import pickle
from datetime import datetime
from vis_exp.vis_bokeh import plotBlokeh


import argparse

argparser =argparse.ArgumentParser()
argparser.add_argument('config_file',help='Train file used ')
argparser.add_argument('layer_name',help='Train file used ')
argparser.add_argument('--cam_label_to_use',help='If not use_pred_max use only this cam class')
argparser.add_argument('--class_to_use',help='Train file used ')
argparser.add_argument('--filter_this_name',help='Train file used ')
argparser.add_argument('--use_pred_max', action='store_true',help='Use cam of max pred')
argparser.add_argument('--activations_file_path',help='Use cam of max pred')
argparser.add_argument('--just_reload_act_file',action='store_true', help='Avoid re calc of activations')
argparser.add_argument('--out_path_dim_red')


args = argparser.parse_args()


config_file = args.config_file
layer_name = args.layer_name
cam_label_to_use = args.cam_label_to_use
class_to_use = args.class_to_use
filter_this_name= args.filter_this_name
use_pred_max = args.use_pred_max

activations_file_path = args.activations_file_path
just_reload_act_file = args.just_reload_act_file
out_path_dim_red = args.out_path_dim_red


if not just_reload_act_file :
    data_config = parse_config_recur(config_file)
    t_params = data_config['train_params']
    m_k = data_config['model_key']
    m_p = data_config['model_params']
    d_k = data_config['dataset_key']
    d_p = data_config['dataset_params']

    if data_config['model_load_path_at_train_file'] is not None:
        model_load_path = data_config['model_load_path_at_train_file']
        print("Using model load path from TRAIN_FILE {0}".format(model_load_path))
    else:
        model_load_path = data_config['model_load_path_train_result']
        print("Using model load path from SELECT_FILE {0}".format(model_load_path))

    batch_size = t_params['b_size'] if 'b_size' in t_params else 20
    epochs = t_params['epochs'] if 'epochs' in t_params else 1


    # open dataset
    model_class = imagenet_classifier_cam_loss
    dataset_class = dataset_obj_dict[d_k]
    base_dataset = dataset_class(epochs, batch_size, **d_p)  # type: Dataset


    # iterate train set

    with model_class(base_dataset,**m_p) as model:  # this also save the train_result
        model.load(model_load_path)

        model.dataset.initialize_iterator_train(model.sess)

        counter=0
        component_mask = {}
        component_mean_v = {}
        all_acts = []

        while True:
            try:
                fd = model.prepare_feed(is_train=False, debug=False)

                # extract activations in batch, CAM
                act_layer,cam_batch,index_batch,soft_max_out,targets = model.sess.run([layer_name,model.cam_out,model.indexs,model.pred,model.targets],feed_dict=fd)


                miss_class = (soft_max_out.argmax(axis=1) != targets.argmax(axis=1))



                # filter
                if use_pred_max:
                    selection = miss_class
                else:
                    selection = np.bitwise_and(miss_class,  (targets.argmax(axis=1)) == class_to_use)# only missclasified

                act_layer = act_layer[selection]
                cam_batch = cam_batch[selection]
                index_batch = index_batch[selection]



                cam_batch = tf.squeeze(cam_batch, axis=[3, 4])



                if use_pred_max:
                    pred_indexs = soft_max_out.argmax(axis=1)[selection]
                    cam_all = cam_batch.eval()
                    sel = [cam_all[i,:,:,pred_indexs[i]] for i in range(len(pred_indexs)) ]
                    cam_batch = np.array(sel)

                else:
                    cam_batch = cam_batch[:,:,:, cam_label_to_use].eval()

                all_acts.append(act_layer.mean(axis=(0,1,2)))

                for i in range(act_layer.shape[0]):

                    index_current = index_batch[i].decode('utf8')

                    # given CAM do a threshold (otsu, percentil)
                    cam_img = cam_batch[i]
                    threshold_global_otsu = threshold_otsu(cam_img)
                    binary_cam = cam_img >= threshold_global_otsu

                    # resize mask if needed
                    if binary_cam.shape != act_layer[i,:,:,0].shape:
                        mask  = cv2.resize(binary_cam, act_layer[i].shape)
                    else:
                        mask = binary_cam

                    # calc conected componets given mask
                    comps = measure.label(mask)
                    n_comps = comps.max()

                    # Mask the activations
                    iters = list(range(n_comps+1))
                    iters.remove(0)


                    for component_index in iters:
                        selected = (comps == component_index)

                        selected_filters = act_layer[i][selected]
                        mean_per_filter = selected_filters.mean(axis=0) # mean per filter

                        # save conected component (index_component,layer_name, index_img,mask_14_14)
                        component_mask[(index_current,component_index)] = selected

                        # save (index_img,index_component,layer_name, mean_vector_component)
                        component_mean_v[(index_current,component_index)] = mean_per_filter


                if counter % 3 == 0:
                    print(counter)
                counter += 1

            except tf.errors.OutOfRangeError:
                log = 'break at {0}'.format(counter)
                break

    mean_per_filt = np.vstack(all_acts).mean(axis=0)


    now = datetime.now()
    now_string = now.strftime('%Y-%b-%d--%H:%M')
    out_path_acts = "dataset_acts_{0}.pkl".format(now_string) if (activations_file_path is None) else activations_file_path
    print('Activations saved to {0}'.format(out_path_acts))


    # save component_mean_v, component_mask, mean_per_filter
    with open("dataset_acts_{0}.pkl".format(now_string),'wb') as f:
        out={'comp_mask' : component_mask, 'comp_mean' : component_mean_v,'mean_p_f' : mean_per_filt }
        pickle.dump(out,f,-1)

else:

    # Open mean_vectors dataset
    print("Loading acts from {0}".format(activations_file_path))
    with open(activations_file_path,'rb') as f:
        out=pickle.load(f)
    component_mask=out['comp_mask']
    component_mean_v=out['comp_mean']
    mean_per_filt=out['mean_p_f']



# NORM HERE ??

keys = list(component_mean_v.keys())
all_data_x = np.vstack([component_mean_v[k] for k in keys])
all_indexs = np.array(['{0}--{1}'.format(k[0],k[1]) for k in keys])


print("Original data shape ",all_data_x.shape)
if filter_this_name:
    out_data_x=[]
    out_indexs=[]
    for ind in range(all_data_x.shape[0]):
        current_ind = all_indexs[ind]
        if filter_this_name in current_ind:
            out_data_x.append(all_data_x[ind])
            out_indexs.append(current_ind)

    all_data_x=np.vstack(out_data_x)
    all_indexs=np.vstack(out_indexs)


print("Original data shape (after filter) ",all_data_x.shape)

# Generate PCA 2D, explained variance,
from sklearn.decomposition import PCA
rd_model = PCA(2)
reduced_d_data_pca = rd_model.fit_transform(all_data_x)
print("Explained variance {0}".format(rd_model.explained_variance_ratio_))

# do TSNE
from sklearn.manifold import TSNE
rd_model=TSNE()
reduced_d_data_tsne = rd_model.fit_transform(all_data_x)


# use UMAP
import umap
reduced_d_data_umap = umap.UMAP().fit_transform(all_data_x)

now = datetime.now()
now_string = now.strftime('%Y-%b-%d--%H:%M')
representation_names = 'representations_{0}'.format(now_string)
out_name = "{0}.pkl".format(representation_names) if out_path_dim_red is None else out_path_dim_red
with open(out_name, 'wb') as f:
    out = {'pca2': reduced_d_data_pca,
           'tsne': reduced_d_data_tsne,
           'umap': reduced_d_data_umap}
    pickle.dump(out, f, -1)

print("Saved to {0}".format(out_name))


