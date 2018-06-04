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


def get_slim_arch_bn(inputs,isTrainTensor,num_classes=1000,scope='vgg_16'):
    """
    from vgg16 https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
    :param inputs:
    :param num_classes:
    :param scope:
    :return:
    """
    with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.

        # Arg scope set default parameters for a list of ops
        with arg_scope([layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                       outputs_collections=end_points_collection):
            net = layers_lib.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
            net = tf.contrib.layers.batch_norm(net,center=True, scale=True,is_training=isTrainTensor,scope='bn1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')

            net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True,is_training=isTrainTensor,scope='bn2')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')

            net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True,is_training=isTrainTensor,scope='bn3')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')

            net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True,is_training=isTrainTensor,scope='bn4')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')

            net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')

            # Here we have 14x14 filters
            net = tf.reduce_mean(net, [1, 2])  # Global average pooling
            net = layers_lib.fully_connected(net, num_classes, activation_fn=None, biases_initializer=None,
                                             scope='softmax_logits')

            # Convert end_points_collection into a end_point dict.
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return net, end_points


class vgg_16_batchnorm(Abstract_model):

    def __init__(self, dataset: Dataset, debug=False,name='vgg16_classifier'):
        super().__init__(dataset, debug,name)

        self.reg_factor = 0.1
        self.lr = 0.001

    def get_feed_dict(self,isTrain):
        return {"phase:0" : isTrain}

    def define_arch(self):
        phase = tf.placeholder(tf.bool, name='phase')

        # Define the model:
        predictions, acts = get_slim_arch_bn(self.input_l,phase,self.dataset.shape_target[0])

        # Configure values for visualization

        self.last_conv = acts['vgg_16/conv5/conv5_3']
        self.softmax_weights = r"vgg_16/softmax_logits/weights:0"
        self.pred = tf.nn.softmax(predictions, name='prediction')

        self.loss = tf.losses.softmax_cross_entropy(self.targets, predictions)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # get accuracy
        prediction = tf.argmax(predictions, 1)
        equality = tf.equal(prediction, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))


if __name__ == '__main__':
    dataset = Imagenet_Dataset(2,40)


    with vgg_16_batchnorm(dataset, debug=False) as model:
        show_graph(tf.get_default_graph())
        save_graph_txt(tf.get_default_graph())
        graph = tf.get_default_graph()
        writer = tf.summary.FileWriter(logdir='logdir', graph=graph)
        writer.flush()
