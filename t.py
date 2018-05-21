import numpy as np
from tensorflow.python.saved_model import tag_constants

from utils import show_graph

a=[1,2,3]
print(np.array(a))

"""
# Install

https://unix.stackexchange.com/questions/332641/how-to-install-python-3-6

# Download data VOC 2017

train val
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

test
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

sudo apt-get install libbz2-dev libncurses5-dev libgdbm-dev liblzma-dev sqlite3 libsqlite3-dev openssl libssl-dev tcl8.6-dev tk8.6-dev libreadline-dev zlib1g-dev      


sudo apt-get install python-setuptools python-dev ncurses-dev


pip install readline

python -m ipykernel install --user --name GSDA_env --display-name "GSDA_env"


# visualizar, seleccionar, generar, ajustar


class model(entrenar / load)
load-model
feed-forward
visualize
train
re_train


(jupyter)
- Dado dataset object, classifier entregar lista de elementos mal clasificados (val set)

- Dado imagen mostrar mapas de clase para cada clase con su porcentaje prob.
- Select interactivo de partes sde imagen
- Guarda lista de mascaras binaria,imagenes a archivo 


find_errors select(img) -> bin_mask
- Genera lista de imagenes evaluando dataset
- Busca imagenes mal clasificadas
- Muestra mapas de visualizacion
- Permite hacer el select



generator(img_list) -> new_img_list
- Con lista de imagenes y mascaras comienza a genear imagenes nuevas
- Entrega un dataset augmentado

re_train
- Dado un objeto dataset.


"""
import tensorflow as tf
import matplotlib.pyplot as plt


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model/check.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))

    print([n.name for n in tf.get_default_graph().as_graph_def().node])

    graph = tf.get_default_graph()
    show_graph(graph)

    test_image = np.random.rand(1,64)

    conv_acts,softmax_w,pred = sess.run(
        ["last_conv_act/Relu:0","softmax_layer/kernel:0","prediction:0"],
        feed_dict={"model_input:0": test_image,"k_prob:0": 1.0 }
    )

    batch_index=0
    index_class=0

    print(conv_acts.shape)
    print(softmax_w.shape)
    print(pred.shape)

    conv_acts = conv_acts[0]
    print("Prediciont {0}".format(pred[batch_index]))

    result = (conv_acts[:, :, :] * softmax_w[:, index_class]).sum(axis=2)



    plt.figure(figsize=(10, 10))
    plt.imshow(test_image.reshape(8, 8))

    plt.figure(figsize=(10, 10))
    plt.imshow(result.reshape(2, 2))

    plt.show()

