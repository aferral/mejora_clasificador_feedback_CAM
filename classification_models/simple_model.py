import pickle

import tensorflow as tf
from datasets.dataset import Dataset
from datasets.cifar10_data import Cifar10_Dataset
from datasets.imagenet_data import Imagenet_Dataset
from datasets.simple_figures import Simple_figures_dataset
from utils import show_graph, save_graph_txt

from classification_models.classification_model import Abstract_model

slim = tf.contrib.slim
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
import numpy as np
import cv2

def get_slim_arch_bn(inputs, isTrainTensor, num_classes=1000, scope='vgg_16'):
    with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.

        filters = 64

        # Arg scope set default parameters for a list of ops
        with arg_scope([layers.conv2d, layers_lib.fully_connected,
                        layers_lib.max_pool2d],
                       outputs_collections=end_points_collection):
            net = layers_lib.repeat(inputs, 2, layers.conv2d, filters, [3, 3],
                                    scope='conv1')
            #p_0 = layers_lib.max_pool2d(net, [2, 2], scope='pool1')


            net = layers_lib.repeat(net, 2, layers.conv2d, filters, [3, 3],
                                    scope='conv2')
            #last_conv = layers_lib.max_pool2d(net, [2, 2], scope='pool2')

            last_conv = net
            # Here we have 14x14 filters
            net = tf.reduce_mean(last_conv, [1, 2])  # Global average pooling

            # add layer with float 32 mask of same shape as global average pooling out
            # feed default with ones, leave placeholder

            mask = tf.placeholder_with_default(tf.ones_like(net),
                                               shape=net.shape, name='gap_mask')
            net = tf.multiply(net, mask)

            net = layers_lib.fully_connected(net, num_classes,
                                             activation_fn=None,
                                             biases_initializer=None,
                                             scope='softmax_logits')

            with tf.variable_scope("raw_CAM"):
                w_tensor_name = "vgg_16/softmax_logits/weights:0"
                s_w = tf.get_default_graph().get_tensor_by_name(w_tensor_name)
                softmax_weights = tf.expand_dims(tf.expand_dims(s_w, 0),
                                                 0)  # reshape to match 1x1xFxC
                # tensor mult from (N x lh x lw x F) , (1 x 1 x F x C)
                cam = tf.tensordot(last_conv, softmax_weights, [[3], [2]],
                                   name='cam_out')

            # Convert end_points_collection into a end_point dict.
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return net, end_points


class simple_classifier(Abstract_model):

    def __init__(self, dataset: Dataset, debug=False,
                 name='simple_classifier',fixed_mask_file=None):
        super().__init__(dataset, debug, name)

        self.reg_factor = 0.1
        self.lr = 0.0001

        if fixed_mask_file:
            with open(fixed_mask_file, 'rb') as f:
                t = pickle.load(f)['masks']
                self.fixed_mask_file = {k : (None,t[k]) for k in t}
        else:
            self.fixed_mask_file = None


    def prepare_feed(self, is_train=False, debug=False):
        if self.fixed_mask_file is None:
            return super().prepare_feed(is_train=is_train,debug=debug)

        else:
            use_simple_index=False
            base_fd = super().prepare_feed(is_train=is_train, debug=True)
            ex_input = base_fd['model_input:0']

            conv_acts, indexs = self.sess.run([self.last_conv, 'indexs_input:0'],base_fd)

            batch_size = ex_input.shape[0]
            h, w, c = conv_acts.shape[1], conv_acts.shape[2], conv_acts.shape[3]

            base_cam_mask = np.zeros((batch_size, h, w))

            for i in range(len(indexs)):
                ind_img = indexs[i].decode('utf-8') if not use_simple_index else indexs[i].decode('utf-8').split('__')[1]
                if ind_img in self.fixed_mask_file:
                    sel_cam, current_mask = self.fixed_mask_file[ind_img]
                    down_sampled = cv2.resize(current_mask.astype(np.float32),
                                              (14, 14))
                    base_cam_mask[i] = down_sampled

            base_fd['cam_loss_term/cam_mask:0'] = base_cam_mask

            return base_fd
        pass

    def get_feed_dict(self, isTrain):
        return {"phase:0": isTrain}

    def define_arch(self):
        phase = tf.placeholder(tf.bool, name='phase')

        # Define the model:
        predictions, acts = get_slim_arch_bn(self.input_l, phase,
                                             self.dataset.shape_target[0])

        self.global_step = tf.Variable(0, trainable=False,name='gstp')


        # Configure values for visualization

        self.last_conv = acts['vgg_16/conv2/conv2_2']
        self.softmax_weights = r"vgg_16/softmax_logits/weights:0"
        self.pred = tf.nn.softmax(predictions, name='prediction')
        self.cam_out = self.graph.get_tensor_by_name('vgg_16/raw_CAM/cam_out:0')

        with tf.variable_scope("cam_loss_term"):

            # nuevo termino de loss = CAM[label](upsample)[mask].sum() * ponderador
            # placeholders
            shape_mask = self.last_conv.shape[0:3]
            cam_mask = tf.placeholder_with_default(tf.zeros_like(self.last_conv[:,:,:,0]),
                                                   shape_mask,
                                                   name='cam_mask')
            self.loss_lambda = tf.placeholder_with_default(0.0, (),
                                                      name='loss_lambda')


            masked_cam = tf.multiply(self.last_conv,tf.expand_dims(tf.cast(cam_mask, dtype=tf.float32), -1),name='masked_cam')

            sum_per_filder = tf.reduce_sum(masked_cam,axis=(1,2))
            acts_per_mask = tf.expand_dims(tf.reduce_sum(cam_mask,axis=(1,2)) + 1 , axis=-1) # add one to avoid divide by zero when no mask

            real_prob = tf.reduce_sum(self.targets * self.pred,axis=1)
            self.act_term = tf.reduce_mean(sum_per_filder / acts_per_mask, axis=1)

            #self.act_loss_term = self.act_term * tf.pow(0.8, real_prob/0.9)
            self.act_loss_term = self.act_term *tf.pow(0.8, real_prob / 0.999)

            self.mean_act_loss_term = tf.reduce_mean(self.act_loss_term)

        with tf.variable_scope("ce_term"):
            ce_term = tf.losses.softmax_cross_entropy(self.targets,predictions)

        self.loss =  ce_term + self.mean_act_loss_term



        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(self.loss,global_step=self.global_step)

        # get accuracy
        prediction = tf.argmax(predictions, 1)
        equality = tf.equal(prediction, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))



if __name__ == "__main__":
    from datasets.quickdraw_dataset import  QuickDraw_Dataset
    with tf.Session().as_default() as sess:
        t = Simple_figures_dataset(100, 60)

        with simple_classifier(t, debug=False) as model:

            model.train(save_model=False)
            model.show_graph()

            model.eval(mode='test')
