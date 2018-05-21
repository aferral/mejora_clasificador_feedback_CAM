from sklearn.datasets import load_digits
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

# From http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
from tensorflow.python.saved_model import tag_constants
from utils import show_graph

class timeit:
    def __enter__(self):
        self.st=time.time()
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('elapsed {0}'.format(time.time()-self.st))
        pass




class clasification_model:
    def prepare_feed(self, iterator, kp=1):
        """
        Feed for iterator in debug=True. This mode doesnt use the iterator directly enabling to use
        sess.run(op) without calling the next batch.
        :param iterator: Iterator object (gives batch_x, batch_y)
        :param kp: keep probability of dropout
        :return: feed_dict for run
        """
        data, labels = iterator.get_next()
        x_batch, y_batch = self.sess.run([data, labels])
        return {self.input_l: x_batch, self.targets: y_batch, self.keep_p: kp}

    def __enter__(self):

        assert (self.sess is None), "Session started"
        self.sess = tf.Session()
        self.sess.as_default()



        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def preprocess_batch(self,image_batch):
        """
        Process a new batch substract train_mean.
        :return:
        """
        return image_batch - self.sess.run('mean_image:0')

    def define_arch(self):
        # Params
        debug = False
        epochs = 20
        reg_factor = 0.1
        batch_size = 30
        kp = 0.95
        lr = 0.001

        # Data load
        digits = load_digits(return_X_y=True)
        # split into train and validation sets
        train_images = digits[0][:int(len(digits[0]) * 0.8)]
        train_labels = digits[1][:int(len(digits[0]) * 0.8)]
        one_hot_train_labels = np.zeros((train_labels.shape[0], 10))
        one_hot_train_labels[np.arange(train_labels.shape[0]), train_labels] = 1

        valid_images = digits[0][int(len(digits[0]) * 0.8):]
        valid_labels = digits[1][int(len(digits[0]) * 0.8):]
        one_hot_val_labels = np.zeros((valid_labels.shape[0], 10))
        one_hot_val_labels[np.arange(valid_labels.shape[0]), valid_labels] = 1



        self.mean = tf.constant(train_images.mean(axis=0), name="mean_image")


        # Create dataset objects
        dx_train = tf.data.Dataset.from_tensor_slices(train_images).map(lambda z: tf.add(z, -self.mean))
        dy_train = tf.data.Dataset.from_tensor_slices(one_hot_train_labels)
        self.train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat(epochs).batch(
            batch_size).cache().prefetch(2000)

        dx_valid = tf.data.Dataset.from_tensor_slices(valid_images).map(lambda z: tf.add(z, -self.mean))
        dy_valid = tf.data.Dataset.from_tensor_slices(one_hot_val_labels)
        self.valid_dataset = tf.data.Dataset.zip((dx_valid, dy_valid)).shuffle(500).repeat(1).batch(
            batch_size).cache().prefetch(2000)

        # Create iterator
        iterator = self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)
        next_element = iterator.get_next()

        # create model
        keep_p = self.keep_p =tf.placeholder(tf.float64, name='k_prob')

        feed_x = tf.placeholder(self.train_dataset.output_types[0], self.train_dataset.output_shapes[0])\
            if debug else next_element[0]
        feed_y = tf.placeholder(self.train_dataset.output_types[1], self.train_dataset.output_shapes[1]) \
            if debug else next_element[1]

        input_l = self.input_l =  tf.placeholder_with_default(feed_x, shape=[None, 64], name='model_input')
        targets = self.targets =  tf.placeholder_with_default(feed_y, shape=[None, 10], name='target')

        inp_reshaped = tf.reshape(input_l, [-1, 8, 8, 1])
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
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_factor), name='softmax_layer')

        self.pred = tf.nn.softmax(out, name='prediction')

        self.loss = tf.losses.softmax_cross_entropy(targets, out)
        self.train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        # get accuracy
        prediction = tf.argmax(out, 1)
        equality = tf.equal(prediction, tf.argmax(targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))





        # Configure feed function.
        self.feed_fun = (lambda: self.prepare_feed(iterator, kp=kp)) if debug else (lambda: {keep_p: kp})
        self.feed_fun_test = (lambda: self.prepare_feed(iterator)) if debug else (lambda: {keep_p: kp})

        # Init ops
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def __init__(self):
        self.sess = None




    def train(self):
        self.define_arch()

        saver = tf.train.Saver()



        training_init_op = self.iterator.make_initializer(self.train_dataset)
        validation_init_op = self.iterator.make_initializer(self.valid_dataset)


        self.sess.run(training_init_op)
        i = 0

        with timeit() as t:
            while True:
                try:
                    fd = self.feed_fun()
                    l, _, acc = self.sess.run([self.loss, self.train_step, self.accuracy], fd)

                    if i % 50 == 0:
                        print("It: {}, loss_batch: {:.3f}, batch_accuracy: {:.2f}%".format(i, l, acc * 100))
                    i += 1
                except tf.errors.OutOfRangeError:
                    print('break at {0}'.format(i))
                    break

        self.sess.run(validation_init_op)
        avg_acc = 0
        c = 0
        while True:
            try:
                fd = self.feed_fun_test()
                acc = self.sess.run([self.accuracy], fd)
                avg_acc += acc[0]
                c += 1
            except tf.errors.OutOfRangeError:
                print(
                    "Average validation set accuracy over {} iterations is {:.2f}%".format(c, (avg_acc / c) * 100))
                break

        # Guardar modelo
        saver.save(self.sess, './model/check')
        graph = tf.get_default_graph()
        show_graph(graph)


    def feed_forward(self):
        pass

    def load(self,metagrap_path,model_folder):
        new_saver = tf.train.import_meta_graph(metagrap_path)
        new_saver.restore(self.sess, tf.train.latest_checkpoint(model_folder))
        graph = tf.get_default_graph()
        show_graph(graph)

    def eval(self):
        pass

    def visualize(self,image):
        """
        Visualize CMAP of image. The image should be a numpy without pre process.
        :param image:
        :return:
        """

        image = self.preprocess_batch( image)


        conv_acts, softmax_w, pred = self.sess.run(
            ["last_conv_act/Relu:0", "softmax_layer/kernel:0", "prediction:0"],
            feed_dict={"model_input:0": image, "k_prob:0": 1.0}
        )

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


if __name__ == '__main__':
    train=False

    with clasification_model() as model:


        if train:
            model.train()
        else:
            model.load('./model/check.meta','./model')


            digits = load_digits(return_X_y=True)
            test_image = (digits[0][1]).reshape(1,64)


            image_processed, prediction, cmaps = model.visualize(test_image)

            test_image_plot = image_processed.reshape((8, 8))


            p_class = np.argmax(prediction)
            print("Predicted {0}".format(p_class))
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
