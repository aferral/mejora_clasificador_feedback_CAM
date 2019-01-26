import tensorflow as tf
from datasets.dataset import Dataset
from datasets.cifar10_data import Cifar10_Dataset
from datasets.imagenet_data import Imagenet_Dataset
from utils import show_graph, save_graph_txt

from classification_models.classification_model import Abstract_model

slim = tf.contrib.slim
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope


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
                                    scope='conv1',
                                    weights_regularizer=slim.l2_regularizer(
                                        0.01))
            bn_0 = tf.contrib.layers.batch_norm(net, center=True, scale=True,
                                                is_training=isTrainTensor,
                                                scope='bn1', decay=0.9)
            p_0 = layers_lib.max_pool2d(bn_0, [2, 2], scope='pool1')

            net = layers_lib.repeat(p_0, 2, layers.conv2d, filters, [3, 3],
                                    scope='conv2',
                                    weights_regularizer=slim.l2_regularizer(
                                        0.01))
            bn_1 = tf.contrib.layers.batch_norm(net, center=True, scale=True,
                                                is_training=isTrainTensor,
                                                scope='bn2', decay=0.9)
            res_1 = p_0 + bn_1
            p_1 = layers_lib.max_pool2d(res_1, [2, 2], scope='pool2')

            net = layers_lib.repeat(p_1, 3, layers.conv2d, filters, [4, 4],
                                    scope='conv3',
                                    weights_regularizer=slim.l2_regularizer(
                                        0.01))
            bn_2 = tf.contrib.layers.batch_norm(net, center=True, scale=True,
                                                is_training=isTrainTensor,
                                                scope='bn3', decay=0.9)
            res_2 = p_1 + bn_2
            p_2 = layers_lib.max_pool2d(res_2, [2, 2], scope='pool3')

            net = layers_lib.repeat(p_2, 3, layers.conv2d, filters, [5, 5],
                                    scope='conv4',
                                    weights_regularizer=slim.l2_regularizer(
                                        0.01))
            bn_3 = tf.contrib.layers.batch_norm(net, center=True, scale=True,
                                                is_training=isTrainTensor,
                                                scope='bn4', decay=0.9)
            res_3 = p_2 + bn_3
            p_3 = layers_lib.max_pool2d(res_3, [2, 2], scope='pool4')

            last_conv = net = layers_lib.repeat(p_3, 3, layers.conv2d, filters,
                                                [5, 5], scope='conv5',
                                                weights_regularizer=slim.l2_regularizer(
                                                    0.01))

            # Here we have 14x14 filters
            net = tf.reduce_mean(net, [1, 2])  # Global average pooling

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


class imagenet_classifier_cam_loss(Abstract_model):

    def __init__(self, dataset: Dataset, debug=False,
                 name='imagenet_classifier'):
        super().__init__(dataset, debug, name)

        self.reg_factor = 0.1
        self.lr = 0.0001

    def get_feed_dict(self, isTrain):
        return {"phase:0": isTrain}

    def define_arch(self):
        phase = tf.placeholder(tf.bool, name='phase')

        # Define the model:
        predictions, acts = get_slim_arch_bn(self.input_l, phase,
                                             self.dataset.shape_target[0])

        # Configure values for visualization

        self.last_conv = acts['vgg_16/conv5/conv5_3']
        self.softmax_weights = r"vgg_16/softmax_logits/weights:0"
        self.pred = tf.nn.softmax(predictions, name='prediction')
        self.cam_out = self.graph.get_tensor_by_name('vgg_16/raw_CAM/cam_out:0')

        with tf.variable_scope("cam_loss_term"):
            sq_cam = tf.squeeze(self.cam_out, axis=[3, 4])  # N x hl x wl x C

            # selecciono solamente CAM del target N x hl x wl x 1
            sel_index = tf.cast(tf.argmax(self.targets, axis=1), tf.int32)
            sel_index = tf.stack([tf.range(tf.shape(sq_cam)[0]), sel_index],
                                 axis=1, name='selected_index')

            # esto es algo complejo pero lo unico que hace es seleccionar por canal el del indice
            selected_cam = tf.gather_nd(tf.transpose(sq_cam, perm=[0, 3, 1, 2]),
                                        sel_index, name='selected_cam')

            # nuevo termino de loss = CAM[label](upsample)[mask].sum() * ponderador
            # placeholders
            cam_mask = tf.placeholder_with_default(tf.zeros_like(selected_cam),
                                                   selected_cam.shape,
                                                   name='cam_mask')
            loss_lambda = tf.placeholder_with_default(0.0, (),
                                                      name='loss_lambda')

            masked_cam = tf.multiply(selected_cam, cam_mask, name='masked_cam')

            cam_loss = tf.multiply(tf.reduce_mean(masked_cam), loss_lambda,
                                   name='loss_cam_v')



        """
        old loss  
        cam_loss = tf.multiply(tf.reduce_sum(masked_cam), loss_lambda,name='loss_cam_v')
        self.loss = tf.losses.softmax_cross_entropy(self.targets,predictions) + cam_loss
        
        # loss lambda ce 
        cam_loss = tf.multiply(tf.reduce_mean(masked_cam), loss_lambda,name='loss_cam_v')
        self.loss = tf.losses.softmax_cross_entropy(self.targets,predictions) * (1+ cam_loss)
        
        loss 3 
        cam_loss = tf.reduce_mean(masked_cam)
        self.loss = tf.losses.softmax_cross_entropy(self.targets,predictions) * (1+ cam_loss)
        """

        self.loss = tf.losses.softmax_cross_entropy(self.targets,predictions) + cam_loss
        # self.loss = cam_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(self.loss)

        # get accuracy
        prediction = tf.argmax(predictions, 1)
        equality = tf.equal(prediction, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))



if __name__ == "__main__":
    from datasets.quickdraw_dataset import  QuickDraw_Dataset
    with tf.Session().as_default() as sess:
        t = QuickDraw_Dataset(1, 60,data_folder='./temp/quickdraw_expanded_images')

        with imagenet_classifier_cam_loss(t, debug=False) as model:
            model.train()
