import time

import tensorflow as tf
import numpy as np
import datetime
import pickle
import cv2
import json

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

def parse_config_recur(config_path):

    all_dicts ={}
    with open(config_path,'r') as f:
        data=json.load(f)

    all_dicts = {**all_dicts,**data}

    t_params = data['train_params']
    gen_file = t_params['gen_file']

    with open(gen_file) as f:
        data_gen=json.load(f)

    all_dicts = {**all_dicts, **data_gen}

    used_select = data_gen['used_select']
    with open(used_select) as f:
        data_select=json.load(f)

    all_dicts = {**all_dicts, **data_select}


    train_result_path = data_select['train_result_path']
    with open(train_result_path,'r') as f:
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