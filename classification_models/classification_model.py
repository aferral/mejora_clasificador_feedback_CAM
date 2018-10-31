from contextlib import ExitStack

import tensorflow as tf
import numpy as np
import os

# From http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
from datasets.dataset import Dataset
from utils import show_graph, now_string, timeit
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def optimistic_restore(session, save_file):
    """
    Restore all the variables in saved file. Usefull for restore a model and add a
    new layers. From https://gist.github.com/iganichev/d2d8a0b1abc6b15d4a07de83171163d4
    :param session:
    :param save_file:
    :return:
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                        var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                            tf.global_variables()),
                        tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)

    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)




class Abstract_model(ExitStack):

    def __init__(self, dataset: Dataset, debug=False, save_name=None):
        super().__init__()
        self.sess = None
        self.dataset = dataset
        self.debug = debug
        self.save_folder=save_name
        self.graph = None
        self.current_log = ''

    def get_name(self):
        return self.save_folder

    def __enter__(self):

        self.graph=self.dataset.graph
        self.enter_context(self.graph.as_default())

        if tf.get_default_session():
            self.sess = tf.get_default_session()
        else:
            self.sess = tf.Session()

        self.enter_context(self.sess.as_default())

        super().__enter__()

        #Build network
        self.define_arch_base()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()
        # DELETE GRAPH
        # tf.reset_default_graph()
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
    def train(self,train_file_used=None,save_model=True,eval=True):

        self.current_log=''
        saver = tf.train.Saver()

        i = 0
        show_batch_dist = True

        with timeit() as t:
            for i in range(1): # todo refactor
                self.dataset.initialize_iterator_train(self.sess)
                while True:
                    try:
                        fd = self.prepare_feed(is_train=True,debug=self.debug)

                        l, _, acc,tgts = self.sess.run([self.loss, self.train_step, self.accuracy,self.targets], fd)


                        if i % 20 == 0:
                            from collections import Counter
                            log ="It: {}, loss_batch: {:.3f}, batch_accuracy: {:.2f}%".format(i, l, acc * 100)
                            self.current_log += '{0} \n'.format(log)
                            print(log)
                            if show_batch_dist:
                                print(Counter(tgts.argmax(axis=1).tolist()))

                        i += 1
                    except tf.errors.OutOfRangeError:
                        log = 'break at {0}'.format(i)
                        self.current_log += '{0} \n'.format(log)
                        print(log)
                        break

                # Eval in val set
                if eval:
                    print("Doing eval")
                    out_string = self.eval()
        # Save model
        if save_model:
            now = now_string()
            path_model_checkpoint = os.path.join('model',self.save_folder,now)
            print("Saving model at {0}".format(path_model_checkpoint))
            os.makedirs(path_model_checkpoint,exist_ok=True)
            saver.save(self.sess, os.path.join(path_model_checkpoint,'saved_model'))

            # save accuracy in val set
            if eval:
                with open(os.path.join(path_model_checkpoint,"accuracy_val.txt"),'w') as f:
                    f.write(out_string)

            # create train_result config
            data = {'mask_files' : [],
                    'model_load_path' : path_model_checkpoint,
                    'train_file_used' : train_file_used}
            out_folder = os.path.join('config_files','train_result')
            os.makedirs(out_folder,exist_ok=True)
            json_name = '{0}__{1}__{2}.json'.format(self.dataset.__class__.__name__,self.__class__.__name__,now)
            with open(os.path.join(out_folder,json_name),'w') as f:
                json.dump(data,f)

            return path_model_checkpoint
        else:
            return None


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

        optimistic_restore(self.sess, tf.train.latest_checkpoint(model_folder))
        # new_saver = tf.train.Saver()
        # new_saver.restore(self.sess, tf.train.latest_checkpoint(model_folder))

        graph = tf.get_default_graph()
        show_graph(graph)

    def eval(self,mode='val',samples=None):

        assert(mode in ['val','train','test']),'Invalid mode choose train, val, test'
        assert((samples is None) or (type(samples) == int)),'Samples must be int or None'
        stop_at = -1 if samples is None else samples

        if mode == 'val':
            name='Validation'
            self.dataset.initialize_iterator_val(self.sess)
        elif mode == 'test':
            name='Test'
            self.dataset.initialize_iterator_test(self.sess)
        else:
            name='Training'
            self.dataset.initialize_iterator_train(self.sess)


        y_true=[]
        y_pred=[]
        iteration=0
        while True:
            try:
                fd = self.prepare_feed(is_train=False,debug=self.debug)
                true,pred = self.sess.run([self.targets,self.pred], fd)
                y_true.append(np.argmax(true,axis=1))
                y_pred.append(np.argmax(pred,axis=1))
                iteration+=1
                if stop_at == iteration:
                    print('Stop at {0}'.format(iteration))
                    break

            except tf.errors.OutOfRangeError:
                break
        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
        out_string = ""
        out_string += "{1} set accuracy is {0:.2f} \n".format(accuracy_score(y_true, y_pred), name)
        out_string += "{0} \n".format(confusion_matrix(y_true, y_pred))
        out_string += "\n \n \n"
        print(out_string)
        return out_string



    def visualize(self,image,norm_cam=True):
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
            if norm_cam:
                temp = (temp - temp.min()) / (temp.max() - temp.min())
            out_maps_per_class[i] = temp

        return img_proc,pred[0],(out_maps_per_class)


class digits_clasifier(Abstract_model):

    def __init__(self, dataset: Dataset, debug=False,name=None):
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


