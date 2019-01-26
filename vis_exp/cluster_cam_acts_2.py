import pickle
from datetime import datetime

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from vis_exp.vis_bokeh import plotBlokeh

repr_file = 'representations_2019-Jan-08--12:04.pkl' #'representations_2018-Dec-11--22:57_all_cm0.pkl'
load_file_path = 'dataset_acts_2019-Jan-08--12:04.pkl' #'dataset_acts_2018-Dec-11--22:39.pkl'

filter_this_name= None #'n02114548'
# select to export mask
select_cluster_label = 2
limit_points = 10
skip_this = []#['n02114855','n02119022']
out_mask_map = {}
out_indexs = []



# open 2d representations and dataset

# Open mean_vectors dataset
with open(load_file_path, 'rb') as f:
    out = pickle.load(f)
component_mask = out['comp_mask']
component_mean_v = out['comp_mean']
mean_per_filt = out['mean_p_f']


keys = list(component_mean_v.keys())
all_data_x = np.vstack([component_mean_v[k] for k in keys])
all_indexs = np.array(['{0}--{1}'.format(k[0],k[1]) for k in keys])

if filter_this_name:
    out_data_x=[]
    out_indexs=[]
    for ind in range(all_data_x.shape[0]):
        current_ind = all_indexs[ind]
        if filter_this_name in current_ind:
            out_data_x.append(all_data_x[ind])
            out_indexs.append(current_ind)

    all_data_x=np.vstack(out_data_x)
    all_indexs=np.vstack(out_indexs).flatten()



with open(repr_file, 'rb') as f:
    out = pickle.load(f)

reduced_d_data_pca= out['pca2']
reduced_d_data_tsne = out['tsne']
reduced_d_data_umap = out['umap']



# plt.scatter(reduced_d_data[:,0],reduced_d_data[:,1])
# plt.show()


# Cluster o link graph para samplear imagen de forma de proponer candidatos
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
cluster_labels_hbscan = clusterer.fit_predict(all_data_x)

# do k means
kmeans = MiniBatchKMeans(n_clusters=10)
cluster_labels_kmeans = kmeans.fit_predict(all_data_x)


# EDIT THIS TO CHANGE REPRESENTATION
to_plot_data=reduced_d_data_tsne
to_plot_labels = cluster_labels_kmeans



print("Original data shape ",all_data_x.shape)
print("Reduced data shape ",to_plot_data.shape)

plotBlokeh(to_plot_data, to_plot_labels,all_indexs , 'test', './')




for index_val in all_indexs[cluster_labels_kmeans == select_cluster_label]:
    index_dataset,component_index = index_val.split("--")
    if any([e in index_dataset for e in skip_this]):
        continue
    out_mask_map[index_dataset] = (None,component_mask[(index_dataset,int(component_index))])
    out_indexs.append(index_dataset)
    if len(out_mask_map) == limit_points:
        break


now = datetime.now()
now_string = now.strftime('%Y-%b-%d--%H:%M')
with open('mask_map_label_{0}__{1}.pkl'.format(select_cluster_label,now_string),'wb') as f:
    pickle.dump(out_mask_map,f,-1)

with open('indexlist_{0}__{1}.txt'.format(select_cluster_label,now_string),'w') as f:
    f.write("\n".join(out_indexs))




