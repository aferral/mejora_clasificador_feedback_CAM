import time

import tensorflow as tf
import numpy as np
import datetime
import pickle
import cv2
import json
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner

def write_imgs_file(img_cam,img_org=None):
    with open('in_buffer_sel.pkl','wb') as f:
        pickle.dump({'img' : img_org,'cam':img_cam},f)
def load_mask():
    with open('out_mask.pkl','rb') as f:
        out=pickle.load(f)
    return out

def now_string():
    a=datetime.datetime.now()
    return a.strftime("%d_%b_%Y__%H_%M")

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def

def data_augmentation(images_tensor : tf.Tensor,size) -> tf.Tensor:
    # from https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
    def flip(x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return x

    def color(x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        return x

    def rotate(x: tf.Tensor) -> tf.Tensor:
        return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4,
                                                   dtype=tf.int32))

    def zoom(x: tf.Tensor) -> tf.Tensor:
        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        scales = list(np.arange(0.8, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))
        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            # Create different crops for an image
            crops = tf.image.crop_and_resize([img], boxes=boxes,
                                             box_ind=np.zeros(len(scales)),
                                             crop_size=(size, size))
            # Return a random crop
            return crops[
                tf.random_uniform(shape=[], minval=0, maxval=len(scales),
                                  dtype=tf.int32)]

        choice = tf.random_uniform(shape=[], minval=0., maxval=1.,
                                   dtype=tf.float32)

        # Only apply cropping 50% of the time
        return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))



    # Add augmentations
    augmentations = [flip, color, zoom, rotate]

    result_tensor = images_tensor

    for f in augmentations:
        result_tensor = tf.cond(tf.random_uniform([], 0, 1) < 0.8, lambda: f(result_tensor),lambda: result_tensor)

    return result_tensor



def parse_config_recur(config_path):

    all_dicts ={}
    with open(config_path,'r') as f:
        data=json.load(f)

    all_dicts = {**all_dicts, **data}

    if 'train_file_used' in data: # is train result file

        data_t_r = data
        path_train_file = data_t_r['train_file_used']

    else: # is train_file with gen

        t_params = data['train_params']
        gen_file = t_params['gen_file']
        with open(gen_file) as f:
            data_gen = json.load(f)
        all_dicts = {**all_dicts, **data_gen}

        used_select = data_gen['used_select']
        with open(used_select) as f:
            data_select = json.load(f)
        all_dicts = {**all_dicts, **data_select}

        train_result_path = data_select['train_result_path']
        with open(train_result_path, 'r') as f:
            data_t_r = json.load(f)
            path_train_file = data_t_r["train_file_used"]


    with open(path_train_file,'r') as f2:
        data_train_file = json.load(f2)

    all_dicts['model_load_path_train_result'] = data_t_r['model_load_path']
    all_dicts['model_load_path_at_train_file'] = data['model_load_path']

    all_dicts = {**all_dicts, **data_train_file}

    return all_dicts


def save_graph_txt(graph):
    with open('graphpb.txt', 'w') as f:
        f.write(str(graph.as_graph_def()))

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))

    with open("test.html",'w') as f:
        f.write(iframe)
    # display(HTML(iframe))


class timeit:
    def __enter__(self):
        self.st=time.time()
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('elapsed {0}'.format(time.time()-self.st))
        pass


def imshow_util_uint8(img_without_preproc,dataset):
    """
    Image from get_index_at (without preprocessing) gets transformed to UINT8 Image for cv2 imshow or plt imshow
    :param img_without_preproc:
    :param dataset:
    :return:
    """
    t=imshow_util(img_without_preproc.reshape(dataset.vis_shape()), dataset.get_data_range())
    return (t*255).astype(np.uint8)


def imshow_util(x,minmaxrange):
    """
    Convert image x to 0-1 float range given the min max original range (for imshow)
    :param x: Image
    :param minmaxrange:
    :return:
    """
    return (x-minmaxrange[0]) / (minmaxrange[1] - minmaxrange[0])


def get_img_cam_index(dataset,classifier,index):
    batch_x, labels = dataset.get_train_image_at(index)
    test_image = batch_x[0]
    r_label = labels[0]
    img = imshow_util(test_image.reshape(dataset.vis_shape()), dataset.get_data_range())

    image_processed, prediction, cmaps = classifier.visualize(test_image)

    all_cams =[]
    for i in range(cmaps.shape[0]):
        im = (cmaps[i]*255).astype(np.uint8)
        t=cv2.resize(im, (img.shape[0], img.shape[1]))
        all_cams.append(t)

    return (img * 255).astype(np.uint8),all_cams,prediction,r_label

def get_img_RAW_cam_index(dataset,classifier,index):


    batch_x, labels = dataset.get_train_image_at(index)
    test_image = batch_x[0]
    r_label = labels[0]
    img = imshow_util(test_image.reshape(dataset.vis_shape()), dataset.get_data_range())

    image_processed, prediction, cmaps = classifier.visualize(test_image,norm_cam=False)

    cmaps=np.array(cmaps)

    all_cams =[]
    for i in range(cmaps.shape[0]):
        cam_i= cmaps[i]
        temp = (cam_i - cam_i.min()) / (cam_i.max() - cam_i.min())
        im = (temp*255).astype(np.uint8)

        t=cv2.resize(im, (img.shape[0], img.shape[1]))
        all_cams.append(t)

    return (img * 255).astype(np.uint8),all_cams,prediction,r_label,cmaps

def get_img_RAW_cam_index_batch(dataset,classifier,index_list,batch_size=100):

    c = 0

    all_imgs = []
    all_cams = []
    all_preds = []
    all_labels = []
    all_raw_cams = []


    while c < len(index_list):
        b_index = index_list[c:c+batch_size]

        tuples_img_labels = [dataset.get_train_image_at(index) for index in b_index]

        b_imgs_raw = np.concatenate(list(map(lambda x : x[0],tuples_img_labels)),axis=0)
        b_r_labels = list(map(lambda x : x[1],tuples_img_labels))

        imgs_out = np.concatenate([ np.expand_dims( imshow_util(img_r.reshape(dataset.vis_shape()),dataset.get_data_range()),axis=0) for img_r in b_imgs_raw], axis=0 )
        imgs_out = (imgs_out * 255).astype(np.uint8)

        imgs_shape = (dataset.vis_shape()[0],dataset.vis_shape()[1])


        img_proc, conv_acts, softmax_w, pred = classifier.feed_forward_vis(b_imgs_raw)

        # Ponderate each last_conv acording to softmax weight and sum channels
        n_c = softmax_w.shape[1]
        b_n = conv_acts.shape[0]
        cams_arr = conv_acts.dot(softmax_w)

        tm = cams_arr.min(axis=(1, 2)).reshape(b_n, 1, 1, n_c)
        tmax = cams_arr.max(axis=(1, 2)).reshape(b_n, 1, 1, n_c)
        norm_cam = (cams_arr - tm) / (tmax - tm)
        cam_to_resize = (norm_cam * 255).astype(np.uint8)

        cam_out = np.concatenate([np.expand_dims(cv2.resize(c, imgs_shape), axis=0) for c in cam_to_resize] , axis=0)

        c += batch_size

        all_imgs.append(imgs_out)
        all_cams.append(cam_out)
        all_preds.append(pred)
        all_labels.append(b_r_labels)
        all_raw_cams.append(cams_arr)

    if len(index_list) > 0:
        all_imgs = np.concatenate(all_imgs,axis=0)
        all_cams = np.concatenate(all_cams,axis=0)
        all_preds = np.concatenate(all_preds,axis=0)
        all_labels = np.concatenate(all_labels,axis=0)
        all_raw_cams = np.concatenate(all_raw_cams,axis=0)

    return all_imgs,all_cams,all_preds, all_labels, all_raw_cams