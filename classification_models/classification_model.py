from contextlib import ExitStack

import tensorflow as tf
import numpy as np
import os

# From http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
from datasets.cifar10_data import Cifar10_Dataset
from datasets.dataset import Dataset, Digits_Dataset
from utils import show_graph, now_string, timeit, do_profile


class Abstract_model(ExitStack):

    def __init__(self, dataset: Dataset, debug=False, save_name=None):
        super().__init__()
        self.sess = None
        self.dataset = dataset
        self.debug = debug
        self.save_folder=save_name

    def get_name(self):
        return self.save_folder

    def __enter__(self):

        if tf.get_default_session():
            self.sess = tf.get_default_session()
        else:
            self.sess = tf.Session()

        super().__enter__()
        self.enter_context(self.sess.as_default())

        #Build network
        self.define_arch_base()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()
        super().__exit__(exc_type, exc_val, exc_tb)

    def define_arch_base(self):

        # Define the inputs
        next_element = self.dataset.get_iterator_entry()
        indexs = self.indexs = tf.placeholder_with_default(next_element[0], shape=next_element[0].shape, name='indexs_input')
        input_l = self.input_l = tf.placeholder_with_default(next_element[1], shape=[None]+self.dataset.shape, name='model_input')
        targets = self.targets = tf.placeholder_with_default(next_element[2], shape=[None]+self.dataset.shape_target, name='target')

        self.define_arch()
        self.check_arch()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def define_arch(self):
        raise NotImplementedError()


    def get_feed_dict(self,isTrain):
        raise NotImplementedError()


    def prepare_feed(self, is_train=False, debug=False):
        """
        Feed for iterator in debug=True. This mode doesnt use the iterator directly enabling to use
        sess.run(op) without calling the next batch.
        :param iterator: Iterator object (gives batch_x, batch_y)
        :param kp: keep probability of dropout
        :return: feed_dict for run
        """

        feed = self.get_feed_dict(is_train)

        if debug:
            inds,x_batch, y_batch = self.sess.run(self.dataset.get_iterator_entry())
            feed[self.indexs] = inds
            feed[self.input_l.name] = x_batch
            feed[self.targets.name] = y_batch
            return feed
        else:
            return feed

    def check_arch(self):
        assert(not (self.loss is None)),'Must define loss '
        assert (not (self.train_step is None)), 'Must define train_step'
        assert (not (self.accuracy is None)), 'Must define accuracy'
        assert (not (self.input_l is None)), 'Must define input_l:'
        assert (not (self.targets is None)), 'Must define targets'

        assert (not (self.last_conv is None)), 'Must define last_conv. Layer before softmax must be conv for visualization'
        assert (not (self.softmax_weights is None)), 'Must define softmax_weights'
        assert (not (self.pred is None)), 'Must define pred. Softmax prediction layer'


    #@do_profile()
    def train(self):


        saver = tf.train.Saver()

        self.dataset.initialize_iterator_train(self.sess)

        i = 0

        with timeit() as t:
            while True:
                try:
                    fd = self.prepare_feed(is_train=True,debug=self.debug)

                    l, _, acc = self.sess.run([self.loss, self.train_step, self.accuracy], fd)


                    if i % 50 == 0:
                        print("It: {}, loss_batch: {:.3f}, batch_accuracy: {:.2f}%".format(i, l, acc * 100))
                    i += 1
                except tf.errors.OutOfRangeError:
                    print('break at {0}'.format(i))
                    break


        # Eval in val set
        self.eval()


        # Save model
        now = now_string()
        path_model_checkpoint = os.path.join('model',self.save_folder,now)
        print("Saving model at {0}".format(path_model_checkpoint))
        os.makedirs(path_model_checkpoint,exist_ok=True)
        saver.save(self.sess, os.path.join(path_model_checkpoint,'saved_model'))

        return path_model_checkpoint


    def feed_forward_vis(self,image):
        image = self.dataset.preprocess_batch(image)

        fd = self.get_feed_dict(isTrain=False)
        fd[self.input_l] = image

        conv_acts, softmax_w, pred = self.sess.run(
            [self.last_conv, self.softmax_weights, self.pred],
            feed_dict=fd
        )

        return image,conv_acts, softmax_w, pred

    def load(self, model_folder):


        new_saver = tf.train.Saver()
        new_saver.restore(self.sess, tf.train.latest_checkpoint(model_folder))

        graph = tf.get_default_graph()
        show_graph(graph)

    def eval(self):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score

        self.dataset.initialize_iterator_val(self.sess)

        y_true=[]
        y_pred=[]
        while True:
            try:
                fd = self.prepare_feed(is_train=False,debug=self.debug)
                true,pred = self.sess.run([self.targets,self.pred], fd)
                y_true.append(np.argmax(true,axis=1))
                y_pred.append(np.argmax(pred,axis=1))

            except tf.errors.OutOfRangeError:
                y_true=np.hstack(y_true)
                y_pred=np.hstack(y_pred)
                print("Validation set accuracy is {:.2f}%".format(accuracy_score(y_true,y_pred)))
                print(confusion_matrix(y_true, y_pred))
                break




    def visualize(self,image):
        """
        Visualize CAM of image. The image should be a numpy without pre process.
        :param image:
        :return:
        """

        if len(image.shape) != (len(self.dataset.shape) +1): # Put in batch format
            image = image.reshape([-1] + list(image.shape))

        img_proc,conv_acts, softmax_w, pred = self.feed_forward_vis(image)

        conv_acts = conv_acts[0]
        # print("conv_acts.shape: {0}".format(conv_acts.shape) )
        # print("softmax_w.shape: {0}".format(softmax_w.shape))
        # print("pred.shape: {0}".format(pred.shape))
        #
        # print("Prediciont {0}".format(pred[0]))
        n_classes = pred.shape[1]
        out_maps_per_class = np.zeros((n_classes,conv_acts.shape[0],conv_acts.shape[0]))

        # Ponderate each last_conv acording to softmax weight and sum channels
        for i in range(n_classes):
            temp = (conv_acts[:,:,:] * softmax_w[:, i]).sum(axis=2)
            # re scale to 0-1
            temp = (temp - temp.min()) / (temp.max() - temp.min())
            out_maps_per_class[i] = temp

        return img_proc,pred[0],(out_maps_per_class)


class digits_clasifier(Abstract_model):

    def __init__(self, dataset: Dataset, debug=False):
        super().__init__(dataset, debug,'digit_classifier')

        self.reg_factor = 0.1
        self.kp = 0.95
        self.lr = 0.001

    def get_feed_dict(self,isTrain):
        return {"k_prob:0": self.kp}

    def define_arch(self):

        # create model
        keep_p = self.keep_p =tf.placeholder(tf.float64, name='k_prob')


        inp_reshaped = tf.cast(tf.reshape(self.input_l, [-1, 8, 8, 1]),tf.float32)
        conv1 = tf.layers.conv2d(inp_reshaped, 30, (2, 2), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2))

        conv2 = tf.layers.conv2d(pool1, 40, (2, 2), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2))

        conv3 = tf.layers.conv2d(pool2, 50, (2, 2), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer()
                                 ,name='last_conv_act')

        global_average_pool = tf.reduce_mean(conv3, [1, 2])

        out = tf.layers.dense(global_average_pool, 10, use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_factor), name='softmax_layer')

        # Configure values for visualization
        self.last_conv = conv3
        self.softmax_weights = "softmax_layer/kernel:0"
        self.pred = tf.nn.softmax(out, name='prediction')

        self.loss = tf.losses.softmax_cross_entropy(self.targets, out)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # get accuracy
        prediction = tf.argmax(out, 1)
        equality = tf.equal(prediction, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))


class cifar10_classifier(Abstract_model):

    def __init__(self, dataset: Dataset, debug=False):
        super().__init__(dataset, debug,'cifar10_classifier')

        self.reg_factor = 0.1
        self.lr = 0.001

    def get_feed_dict(self,isTrain):
        return {}

    def define_arch(self):

        r_input = tf.cast(self.input_l, tf.float32)
        conv1 = tf.layers.conv2d(r_input, 30, (3, 3), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2))

        conv2 = tf.layers.conv2d(pool1, 40, (3, 3), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2))

        conv3 = tf.layers.conv2d(pool2, 50, (3, 3), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer()
                                 ,name='last_conv_act')

        global_average_pool = tf.reduce_mean(conv3, [1, 2])

        out = tf.layers.dense(global_average_pool, 10, use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_factor), name='softmax_layer')

        # Configure values for visualization
        self.last_conv = conv3
        self.softmax_weights = "softmax_layer/kernel:0"
        self.pred = tf.nn.softmax(out, name='prediction')

        self.loss = tf.losses.softmax_cross_entropy(self.targets, out)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # get accuracy
        prediction = tf.argmax(out, 1)
        equality = tf.equal(prediction, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))


class CWR_classifier(Abstract_model):

    def __init__(self, dataset: Dataset, debug=False):
        super().__init__(dataset, debug,'CWR_Clasifier')

        self.reg_factor = 0.1
        self.lr = 0.001

    def get_feed_dict(self,isTrain):
        return {}

    def define_arch(self):

        # inp_reshaped = tf.reshape(self.input_l, [-1, 8, 8, 1])
        conv1 = tf.layers.conv2d(self.input_l, 30, (3, 3), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2))

        conv2 = tf.layers.conv2d(pool1, 40, (3, 3), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2))

        conv3 = tf.layers.conv2d(pool2, 50, (3, 3), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer()
                                 ,name='last_conv_act')

        global_average_pool = tf.reduce_mean(conv3, [1, 2])

        out = tf.layers.dense(global_average_pool, 4, use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_factor), name='softmax_layer')

        # Configure values for visualization
        self.last_conv = conv3
        self.softmax_weights = "softmax_layer/kernel:0"
        self.pred = tf.nn.softmax(out, name='prediction')

        self.loss = tf.losses.softmax_cross_entropy(self.targets, out)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # get accuracy
        prediction = tf.argmax(out, 1)
        equality = tf.equal(prediction, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))



def imshow_util(x,minmaxrange):
    """
    Convert image x to 0-1 float range given the min max original range (for imshow)
    :param x: Image
    :param minmaxrange:
    :return:
    """
    return (x-minmaxrange[0]) / (minmaxrange[1] - minmaxrange[0])

