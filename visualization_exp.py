import numpy as np
import tensorflow as tf
import os

from classification_models.classification_model import imshow_util, \
    CWR_classifier
from classification_models.vgg16_edited import vgg_16_CAM
from classification_models.vgg_16_batch_norm import vgg_16_batchnorm
from datasets.cifar10_data import Cifar10_Dataset
from datasets.cwr_dataset import CWR_Dataset
from datasets.imagenet_data import Imagenet_Dataset
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.util import invert

from utils import timeit, do_profile, now_string

"""

1.Son estas visualizaciones buenas para localizar areas de interes en la red ??

Es decir de cambiar estos pixeles de interes afecta la clasificacion considerablemente. Hipotesis nula
cambiando otras areas afecta de igual forma

    Correspondencia con oclusion localizada
        - Insertar cuadrado gris / promedio / ruido
        - Ir moviendo por diversas areas y calcular probabilidad de clase predicha
        - Crear mapa de calor con esto

        - Evaluar mascara binaria entre este y el otro. Metrica total por dataset.

** For iterator
    - calcular mejor prediccion = best_class
    - Calcular CAM correspondiente best_class
    -Iniciar imagen_out
    - dado cuadraro de 0.1 de imagen size
    - definir batch size

    for posx,posy in posibles:
        - Sobreponer el patron en posiciones posibles (area / promeido / ruido)
        - cuando se llene batch propagar
        - anotar en imagen_out[px,py] prob de best_class

    - Observar ambas iamgen de CAM y generada con probs de este
    - Binarizar y comparar. Usar otsu o 90% percentil y comparar overlap de imagenes
    Metrica de error por imagen
    
    
    
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.util import invert


norm_o=(oclusion_maps[ind] - oclusion_maps[ind].min()) / (oclusion_maps[ind].max() - oclusion_maps[ind].min())
norm_o=(norm_o*2) - 1
norm_o = invert(norm_o)
norm_o = resize(norm_o,mask_dim)


i=5
norm_cam=(cam_maps[i] - cam_maps[i].min()) / (cam_maps[i].max() - cam_maps[i].min())
norm_cam=(norm_cam*2) - 1
norm_cam = resize(norm_cam,mask_dim)


# Mascara binaria del 70% superior
bin_cam = norm_cam > 0.4
plt.figure()
plt.imshow(bin_cam)

# Mascara binaria del 70% superior
bin_oc = alt_oclusions[i] > 0.4
plt.figure()
plt.imshow(bin_oc)


plt.figure()
plt.imshow(im_batch[i])

plt.figure()
plt.imshow(oclusion_maps[i])

plt.figure()
plt.imshow(norm_cam)


plt.figure()
plt.imshow(alt_oclusions[i])
plt.show()

    

"""

def random_pattern_test(classifier,fd,pred_class,base_img,binary_mask,samples=25):

    dims = binary_mask.shape
    assert(len(dims) == 2)

    randoms=np.random.rand(samples,2)

    imgs = []

    # move binary_mask
    for i in range(samples):
        nm=np.roll(np.roll(binary_mask, int(randoms[i, 1] * dims[1]), axis=1),int(randoms[i, 0] * dims[0]), axis=0)
        aplied_mask = base_img.copy()
        aplied_mask[nm] = 0
        imgs.append(aplied_mask)

    aplied_mask = base_img.copy()
    aplied_mask[binary_mask] = 0
    imgs.append(aplied_mask)

    # Calc probs
    fd['model_input:0'] = np.stack(imgs,axis=0)
    oc_pred = classifier.sess.run(classifier.pred, fd)[:, pred_class]

    # calc std and mean
    return oc_pred[-1], oc_pred[:-1].mean(), oc_pred[:-1].std()


# @do_profile()
def exp_CAM_eval(dataset,classifier,out_folder):

    dataset.initialize_iterator_val(classifier.sess)

    mask_dim = dataset.shape if len(dataset.shape) == 2 else dataset.shape[0:2]

    all_probs=[]
    all_std_dists=[]

    counter =0
    with timeit():
        while True:
            try:
                fd = classifier.prepare_feed(is_train=False, debug=True)

                # Calc prediction, target, conv_acts
                tensors=[classifier.last_conv, classifier.softmax_weights, classifier.targets, classifier.pred]
                conv_acts,softmax_w,y_real ,y_pred = classifier.sess.run(tensors, fd)

                pred_class = np.argmax(y_pred,axis=1)
                pred_values = np.max(y_pred,axis=1)

                batch_s = conv_acts.shape[0]
                n_filters = softmax_w.shape[0]

                # Calc CAM of predictions
                # np.array_equal( softmax_w[:,pred_class][:,2], softmax_w[:,pred_class[2]])

                predicted_soft_w = softmax_w[:,pred_class]
                predicted_soft_w = predicted_soft_w.T.reshape(batch_s, 1, 1, n_filters) # (for broadcasting in the filter h,w dimension)

                cam_maps = (conv_acts * predicted_soft_w).sum(axis=-1) # Element wise multiplication per channel and then sum all the channels for a batch
                # equivalent to np.array_equal((conv_acts[1,:, :, :] * softmax_w[:, pred_class[1]]).sum(axis=2), res[1])

                im_batch = fd['model_input:0']
                for ind,image in enumerate(im_batch):
                    norm_cam = (cam_maps[ind] - cam_maps[ind].min()) / (cam_maps[ind].max() - cam_maps[ind].min())
                    norm_cam = (norm_cam * 2) - 1 # to -1 +1 range
                    norm_cam = resize(norm_cam, mask_dim,mode='constant')

                    binary_mask = norm_cam > 0.6

                    prob,mean,std = random_pattern_test(classifier, fd, pred_class[ind],image, binary_mask,samples=25)

                    diff_prob = prob - pred_values[ind]
                    std_dist = abs(prob-mean)*1.0 / std if std != 0 else 0

                    all_probs.append(diff_prob)
                    all_std_dists.append(std_dist)

                    counter+=1
                    if counter % 100 == 0:
                        print("It: {0}".format(counter))


            except Exception as err:
                print(str(err))
                # Filter outliers segun IQR de 4 std fuera
                all_std_dists = np.array(all_std_dists)
                q75, q25 = np.percentile(all_std_dists, [75, 25])
                iqr = q75 - q25
                min = q25 - (iqr * 1.5)
                max = q75 + (iqr * 1.5)
                filtered_std_dists = all_std_dists[np.bitwise_and(all_std_dists > min, all_std_dists < max)]

                mean_diff_prob = np.array(all_probs).mean()
                mean_std_dist = filtered_std_dists.mean()

                print("Images used for calculation: {0}".format(len(all_probs)))
                print("Mean_prob_change : {0}, Mean_STD_distance: {1}".format(mean_diff_prob, mean_std_dist))

                f,axes=plt.subplots(1,2)
                axes[0].boxplot(filtered_std_dists)
                axes[0].set_title("Std distances")
                axes[1].boxplot(np.array(all_probs))
                axes[1].set_title("Diff prob")

                # write results to json
                import json
                out_path = os.path.join(out_folder, "result.json")
                with open(out_path,'w') as file:
                    json.dump({'mean_diff_prob' : float(mean_diff_prob), "mean_std_dist" : float(mean_std_dist) },file)

                # write boxplot figures
                out_path = os.path.join(out_folder,"result.png")
                plt.savefig(out_path)

                break


def visualize_dataset_CAM_predicted(dataset,model,out_folder):

    dataset.initialize_iterator_val(model.sess)

    # make imgs folder
    imgs_f = os.path.join(out_folder,"CAM_imgs")
    os.makedirs(imgs_f,exist_ok=True)

    counter=0
    with timeit():
        while True:
            try:
                fd = model.prepare_feed(is_train=False, debug=False)

                # Calc prediction, target, conv_acts
                tensors=[model.indexs,model.input_l,model.last_conv, model.softmax_weights, model.targets, model.pred]
                indexs,images,conv_acts,softmax_w,y_real ,y_pred = model.sess.run(tensors, fd)

                pred_class = np.argmax(y_pred,axis=1)
                pred_values = np.max(y_pred,axis=1)

                batch_s = conv_acts.shape[0]
                n_filters = softmax_w.shape[0]

                # Calc CAM of predictions
                # np.array_equal( softmax_w[:,pred_class][:,2], softmax_w[:,pred_class[2]])

                predicted_soft_w = softmax_w[:,pred_class]
                predicted_soft_w = predicted_soft_w.T.reshape(batch_s, 1, 1, n_filters) # (for broadcasting in the filter h,w dimension)

                cam_maps = (conv_acts * predicted_soft_w).sum(axis=-1) # Element wise multiplication per channel and then sum all the channels for a batch
                # equivalent to np.array_equal((conv_acts[1,:, :, :] * softmax_w[:, pred_class[1]]).sum(axis=2), res[1])

                for ind,img in enumerate(images):
                    img = dataset.inverse_preprocess(img)
                    test_image_plot = imshow_util( img.reshape(dataset.vis_shape()),dataset.get_data_range())

                    p_class = pred_class[ind]

                    out_shape = list(test_image_plot.shape)
                    if len(test_image_plot.shape) == 3:
                        out_shape = out_shape[0:2]

                    cam_img = imshow_util(cam_maps[ind],[cam_maps[ind].min(),cam_maps[ind].max()])
                    resized_map = resize(cam_img,out_shape,mode='constant')

                    fig, ax = plt.subplots()
                    ax.set_title("Predicted {0} with score {1}".format(p_class,pred_values[ind]))
                    ax.imshow(resized_map, cmap='jet',alpha=0.5)
                    ax.imshow(test_image_plot,alpha=0.5,cmap='gray')

                    img_out_path = os.path.join(imgs_f,"{0}.png".format(indexs[ind]))
                    plt.savefig(img_out_path)
                    if (counter % 100 == 0):
                        print("Image {0}".format(counter))
                    counter+=1


            except tf.errors.OutOfRangeError as err:
                print("Ended")
                break



"""

Necesito acceso imagen = id en tiempo constante

**Pasada  dataset clasificando retorna
id, class_scores, real_class, predicted

**Pasada filter bad
id, class_scores, real_class, predicted

**vis(indice)




2. Analisis de trozos.

2.1 Existe consistencia en los trozos correctamente clasificados??
** For iterator
- calcular pred,real
- SI es pred == pred seguir
- Calcular CAM clase pred
- Binarizar y cortar segun mascara. Guardar a carpeta


2.2 Existe consistencia en los trozos incorrectamente clasificados de una clase??
** For iterator
- calcular pred,real
- SI es pred != pred seguir
- Calcular CAM clase pred
- Calcular CAM clase pred
- Binarizar y cortar segun mascara. Guardar a carpeta ambas imagenes



"""

if __name__ == '__main__':


    # dataset = Imagenet_Dataset(20,30)
    # dataset = Cifar10_Dataset(20,40)
    dataset = CWR_Dataset(4,60)

    with CWR_classifier(dataset, debug=False) as model:
        out_path_folder = os.path.join('vis_results',model.get_name() + "_" + now_string())
        os.makedirs(out_path_folder,exist_ok=True)

        # model.load('.','model/Imagenet_subset_vgg16_CAM/29_May_2018__15_16')
        # model.load('./model/check.meta','./model/vgg16_classifier/29_May_2018__01_41')
        model.load('./model/CWR_Clasifier/14_Jun_2018__13_36')

        # exp_CAM_eval(dataset, model,out_path_folder)
        visualize_dataset_CAM_predicted(dataset, model, out_path_folder)