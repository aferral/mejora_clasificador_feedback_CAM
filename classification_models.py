from sklearn.datasets import load_digits
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

# From http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
from tensorflow.python.saved_model import tag_constants

from dataset import Dataset, Digits_Dataset
from utils import show_graph

class timeit:
    def __enter__(self):
        self.st=time.time()
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('elapsed {0}'.format(time.time()-self.st))
        pass


class clasification_model:

    def __init__(self,dataset : Dataset,debug=False):
        self.sess = None
        self.dataset = dataset
        self.debug = debug



    def __enter__(self):
        assert (self.sess is None), "A session is already active"
        self.sess = tf.Session()
        self.sess.as_default()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def define_arch_base(self):

        # Define the inputs
        next_element = self.dataset.get_iterator_entry()
        input_l = self.input_l =  tf.placeholder_with_default(next_element[0], shape=[None]+self.dataset.shape, name='model_input')
        targets = self.targets =  tf.placeholder_with_default(next_element[1], shape=[None]+self.dataset.shape_target, name='target')

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
            x_batch, y_batch = self.sess.run(self.dataset.get_iterator_entry())
            feed[self.input_l] = x_batch
            feed[self.targets] = y_batch
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



    def train(self):
        self.define_arch_base()


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
        saver.save(self.sess, './model/check') # todo config output folder



    def feed_forward_vis(self,image):
        image = self.dataset.preprocess_batch(image)

        fd = self.get_feed_dict(isTrain=False)
        fd[self.input_l] = image

        conv_acts, softmax_w, pred = self.sess.run(
            [self.last_conv, self.softmax_weights, self.pred],
            feed_dict=fd
        )

        return conv_acts, softmax_w, pred

    def load(self,metagrap_path,model_folder):


        self.define_arch_base()

        new_saver = tf.train.Saver()
        new_saver.restore(self.sess, tf.train.latest_checkpoint(model_folder))

        graph = tf.get_default_graph()
        show_graph(graph)

    def eval(self):

        self.dataset.initialize_iterator_val(self.sess)

        avg_acc = 0
        c = 0
        while True:
            try:
                fd = self.prepare_feed(is_train=False,debug=self.debug)
                acc = self.sess.run([self.accuracy], fd)
                avg_acc += acc[0]
                c += 1
            except tf.errors.OutOfRangeError:
                print(
                    "Average validation set accuracy over {} iterations is {:.2f}%".format(c, (avg_acc / c) * 100))
                break


    def visualize(self,image):
        """
        Visualize CAM of image. The image should be a numpy without pre process.
        :param image:
        :return:
        """

        conv_acts, softmax_w, pred = self.feed_forward_vis(image)

        conv_acts = conv_acts[0]
        print("conv_acts.shape: {0}".format(conv_acts.shape) )
        print("softmax_w.shape: {0}".format(softmax_w.shape))
        print("pred.shape: {0}".format(pred.shape))


        print("Prediciont {0}".format(pred[0]))
        n_classes = pred.shape[1]
        out_maps_per_class = np.zeros((n_classes,conv_acts.shape[0],conv_acts.shape[0]))

        # Ponderate each last_conv acording to softmax weight and sum channels
        for i in range(n_classes):
            temp = (conv_acts[:,:,:] * softmax_w[:, i]).sum(axis=2)
            # re scale to 0-1
            temp = (temp - temp.min()) / (temp.max() - temp.min())
            out_maps_per_class[i] = temp

        return image,pred[0],(out_maps_per_class)


class digits_clasifier(clasification_model):

    def __init__(self, dataset: Dataset, debug=False):
        super().__init__(dataset, debug)

        self.reg_factor = 0.1
        self.kp = 0.95
        self.lr = 0.001

    def get_feed_dict(self,isTrain):
        return {"k_prob:0": self.kp}

    def define_arch(self):

        # create model
        keep_p = self.keep_p =tf.placeholder(tf.float64, name='k_prob')


        inp_reshaped = tf.reshape(self.input_l, [-1, 8, 8, 1])
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








if __name__ == '__main__':
    train=True

    # Todo improve model save
    # todo improve dataset get
    # start to work on mnist or VOC

    dataset = Digits_Dataset(epochs=20,batch_size=30)

    with digits_clasifier(dataset, debug=False) as model:

        if train:
            model.train()
        else:
            model.load('./model/check.meta','./model')


            # todo imporve this
            digits = load_digits(return_X_y=True)
            test_image = (digits[0][1]).reshape(1,64)


            image_processed, prediction, cmaps = model.visualize(test_image)

            test_image_plot = image_processed.reshape((8, 8))


            p_class = np.argmax(prediction)
            print("Predicted {0} with score {1}".format(p_class,np.max(prediction)))
            print(cmaps.shape)
            print("CMAP: ")

            import matplotlib.pyplot as plt
            from skimage.transform import resize

            plt.figure()
            plt.imshow(test_image.reshape(8,8),cmap='gray')

            plt.figure()
            plt.imshow(test_image_plot,cmap='gray')


            plt.figure()
            plt.imshow(cmaps[0],cmap='jet',interpolation='none')


            resized_map = resize(cmaps[0],(test_image_plot.shape))
            plt.figure()
            plt.imshow(resized_map,cmap='jet')

            fig, ax = plt.subplots()
            ax.imshow(resized_map, cmap='jet',alpha=0.6)
            ax.imshow(test_image_plot,alpha=0.4,cmap='gray')
            plt.show()
