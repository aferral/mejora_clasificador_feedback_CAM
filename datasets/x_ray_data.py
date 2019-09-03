import pickle

import cv2
import tensorflow as tf
import os
from datasets.dataset import Dataset
from tf_records_parser.cifar10 import LOCAL_FOLDER
import numpy as np
import json

from utils import data_augmentation

IMAGE_SHAPE = [224, 224, 1]

def parse_function(example_proto):
    features = {"image_raw": tf.FixedLenFeature((), tf.string),
                "label": tf.FixedLenFeature((), tf.int64),
                "height": tf.FixedLenFeature((), tf.int64),
                "width": tf.FixedLenFeature((), tf.int64),
                "depth": tf.FixedLenFeature((), tf.int64),
                "index": tf.FixedLenFeature((), tf.string)
                }

    parsed_features = tf.parse_single_example(example_proto, features)
    h, w, d = parsed_features['height'], parsed_features['width'], parsed_features['depth']

    flat_image = tf.decode_raw(parsed_features["image_raw"], tf.uint8)
    reconst = tf.cast(tf.transpose(tf.reshape(flat_image, tf.stack([d, h, w])), [1, 2, 0], 'reconstructed_image'),tf.float32)

    return parsed_features["index"],reconst, parsed_features["label"]



def get_n_classes():
    return 4

def get_records_folder():
    return os.path.join('temp', "dataset_xray_tfrecords")



class Xray_dataset(Dataset):

    def __init__(self, epochs, batch_size, data_folder=None, **kwargs):


        self.n_classes = get_n_classes()
        print("Loaded {0} classes".format(self.n_classes))

        if data_folder:
            self.data_folder = data_folder

            with open(os.path.join(data_folder, 'labelnames.json'), 'r') as f:
                full_data = json.load(f)
                self.index_dict = full_data['indexs']
                self.names_dict = full_data['names']
                self.label_dict = {self.names_dict[k] : k for k in self.names_dict}


        train_records = list(filter(lambda x : 'train' in x , os.listdir(get_records_folder())))
        val_records = list(filter(lambda x : 'val' in x , os.listdir(get_records_folder())))
        test_records = list(filter(lambda x : 'test' in x , os.listdir(get_records_folder())))

        train_records = list(map(lambda x : os.path.join(get_records_folder(),x),train_records))
        val_records = list(
            map(lambda x: os.path.join(get_records_folder(), x), val_records))
        test_records = list(
            map(lambda x: os.path.join(get_records_folder(), x), test_records))




        dataset_train = tf.data.TFRecordDataset(train_records).map(parse_function)
        dataset_val = tf.data.TFRecordDataset(val_records).map(parse_function)
        dataset_test = tf.data.TFRecordDataset(test_records).map(parse_function)

        # Calculate mean image, std image
        mean_image_path = os.path.join(LOCAL_FOLDER, 'mean_xray.npy')

        if not os.path.exists(mean_image_path):
            sess = tf.get_default_session()
            temp_iterator = dataset_train.batch(10).make_one_shot_iterator().get_next()
            p_mean = np.zeros(IMAGE_SHAPE)
            c = 0

            try:
                while True:
                    index, batch_x, batch_y = sess.run(temp_iterator)
                    p_mean = p_mean + np.mean(batch_x, axis=0)  # IMAGE_SHAPE
                    c += 1
            except tf.errors.OutOfRangeError:
                self.mean = (p_mean / c).astype(np.float32)
                np.save(mean_image_path, self.mean)
                print("Saved mean image")
        else:
            print("Mean image loaded from file")
            print("Train index list loaded from file")
            self.mean = np.load(mean_image_path)



        self.train_index_list = list(filter(lambda x : 'train' in x , self.index_dict.keys()))
        # todo test_code
        # self.train_index_list = list(self.index_dict.keys())


        def preprocess(index, x, y):
            return (index, tf.add(tf.cast(x,tf.float32), -self.mean) / 255, tf.one_hot(y, self.n_classes))

        # .apply(tf.contrib.data.map_and_batch( map_func=preprocess, batch_size=batch_size))
        # .cache()
        # .map(preprocess,num_parallel_calls=4).batch(batch_size)
        # .prefetch(1)
        # .shuffle(10)


        # preprocesss
        shuffle_buffer = int(self.n_classes*300*0.8)
        self.train_dataset = dataset_train.map(preprocess, num_parallel_calls=4).shuffle(shuffle_buffer).cache().repeat(epochs).batch(
            batch_size).prefetch(3)
        self.valid_dataset = dataset_val.map(preprocess, num_parallel_calls=4).shuffle(shuffle_buffer).cache().repeat(1).batch(
            batch_size).prefetch(3)
        self.dataset_test = dataset_test.map(preprocess, num_parallel_calls=4).shuffle(shuffle_buffer).cache().repeat(1).batch(
            batch_size).prefetch(3)

        # Create iterator
        iterator = self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                                   self.train_dataset.output_shapes)

        # check parameters
        super().__init__(**kwargs)

    def preprocess_batch(self, image_batch):
        """
        Process a new batch substract train_mean.
        :return:
        """

        return (image_batch - self.mean) / 255

    def inverse_preprocess(self, image_batch):
        return (image_batch * 255) + self.mean

    @property
    def shape(self):
        return IMAGE_SHAPE

    @property
    def shape_target(self):
        return [self.n_classes]

    def get_index_list(self):
        assert (hasattr(self, 'data_folder')), "Image folder undefined"
        return self.train_index_list

    def get_train_image_at(self, index,strict=False):  # index is image path

        assert (hasattr(self, 'data_folder')), "Image folder undefined"
        if not('train' in index):
            print("WARNING IMAGE NOT IN TRAIN LIST YOU ARE USING A TEST OR VALIDATION IMAGE {0}".format(index))
            if strict:
                print("Strict mode returning NONE")
                return None
        full_path = self.index_dict[index]
        class_name = full_path.split(os.sep)[-2]
        img = cv2.imread(full_path)
        s = self.vis_shape()
        img_out = cv2.resize(img, tuple(s[0:2]))  # original image need resize

        img_out_rgb=cv2.cvtColor(img_out,cv2.COLOR_BGR2GRAY) # convert image from BGR to GRAY

        return img_out_rgb.reshape(1, s[0], s[1], 1), [self.names_dict[class_name]]

    def get_data_range(self):
        return [0, 255]

    def vis_shape(self):
        return IMAGE_SHAPE[:2]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    with tf.Session().as_default() as sess:
        t = Xray_dataset(1, 10,data_folder='./temp/dataset_xray_imgs')
        img,label=t.get_train_image_at('train_44')
        plt.imshow(img.squeeze())
        plt.show()
