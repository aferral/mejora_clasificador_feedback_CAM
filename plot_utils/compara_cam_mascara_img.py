import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

ind_labels = {"n02114548_5207.JPEG" : 0,
"n02114548_10505.JPEG" : 0,
"n02114548_11513.JPEG" : 0,
"n02120079_4409.JPEG" : 3,
"n02120079_9808.JPEG" : 3}


pb = '/home/aferral/PycharmProjects/generative_supervised_data_augmentation/out_backprops/loss_v2_few_0_cam_V2_loss_07_May_2019__11_38'
path_or = os.path.join(pb,'originals')
files_in_or = os.listdir(path_or)

data={}
for elem in files_in_or:
    is_mask = 'mask' in elem
    if is_mask:
        ind = elem.replace('_mask','').replace('.png','')
    else:
         ind = elem.replace('.png','')

    if is_mask:
        data.setdefault(ind,{})['mask'] = os.path.join(path_or,elem)
    else:
        data.setdefault(ind,{})['img'] = os.path.join(path_or,elem)

# consigue mascara, consigue indice

# abre last_raw_cam de indice
for ind in data:
    f_nane = "real_{0}_raw_vis_it_100.npy".format(ind)
    path_raw_cam = os.path.join(pb,'raw_cams',f_nane)
    label =ind_labels[ind]
    raw_data = np.load(path_raw_cam)
    sel = raw_data[label]
    r_cam = cv2.resize(sel,(224,224))
    img_d = cv2.imread(data[ind]['img'])
    mask = cv2.imread(data[ind]['mask'])

    red  = cv2.applyColorMap((((r_cam-r_cam.min())/(r_cam.max()-r_cam.min()))*255).astype(np.uint8), cv2.COLORMAP_JET)


    bin_mask = mask.mean(axis=2) > 0

    res=cv2.addWeighted(img_d, 0.5, red, 0.5, 0)
    res[bin_mask] = [255,0,0]

    plt.imshow(cv2.cvtColor(res,cv2.COLOR_BGR2RGB));plt.show()
    cv2.imwrite('res_{0}.png'.format(ind),res)



