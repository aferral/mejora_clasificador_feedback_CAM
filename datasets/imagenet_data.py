import pickle

import cv2
import tensorflow as tf
import os
from datasets.dataset import Dataset
from tf_records_parser.cifar10 import LOCAL_FOLDER
import numpy as np
import json

from utils import data_augmentation

IMAGE_SHAPE = [224, 224, 3]


def parse_function(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string),
                "image/label": tf.FixedLenFeature((), tf.int64),
                "image/height": tf.FixedLenFeature((), tf.int64),
                "image/width": tf.FixedLenFeature((), tf.int64),
                "image/channels": tf.FixedLenFeature((), tf.int64),
                "index": tf.FixedLenFeature((), tf.string)

                }
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_jpeg(parsed_features["image/encoded"], channels=3)

    # tf.Assert(tf.shape(image)[2] == 3,[image])
    reconst = tf.image.resize_images(image, IMAGE_SHAPE[0:2])

    reconst = tf.cast(reconst, tf.float32, name='image_reshaped')

    return parsed_features["index"], reconst, parsed_features["image/label"]


def get_n_classes():
    class_map = os.path.join(LOCAL_FOLDER, "tfrecords_imagenet_subset", "labelnames.json")
    with open(class_map, 'r') as f:
        names_data = json.load(f)
    return len(names_data)
def get_records_folder():
    return os.path.join(LOCAL_FOLDER, "tfrecords_imagenet_subset")


def list_records():
    path_records = get_records_folder()

    # list all tfrecords
    tfrecords = list(map(lambda x: os.path.join(path_records, x), sorted(filter(
        lambda name: name.split('.')[-1] == 'tfrecord',
        os.listdir(path_records)))))
    return tfrecords


class Imagenet_Dataset(Dataset):

    def __init__(self, epochs, batch_size, data_folder=None,use_all_in_train=False,add_data_augmentation=False, **kwargs):
        tfrecords = list_records()
        n_records = len(tfrecords)
        self.n_classes = get_n_classes()
        print("Loaded {0} classes".format(self.n_classes))

        if data_folder:
            self.data_folder = data_folder

            # load class_name -> label map
            class_map = os.path.join(self.data_folder,"labelnames.json")
            with open(class_map,'r') as f:
                names_data=json.load(f)
            self.class_name_map = {class_name : ind for (ind,class_name) in names_data.values()}



        train_n_records = int(0.8 * n_records)
        val_n_records = int(0.1 * n_records) if int(0.1 * n_records) > 0 else 1
        test_n_records = n_records - train_n_records - val_n_records
        assert (train_n_records != 0 and val_n_records != 0 and test_n_records != 0)

        if use_all_in_train:
            train_records = tfrecords
        else:
            assert (train_n_records + val_n_records + test_n_records == n_records), "The train-val-test split must use all tf records."
            train_records = tfrecords[0:train_n_records]

        validation_records = tfrecords[train_n_records:train_n_records + val_n_records]
        test_records = tfrecords[train_n_records + val_n_records:]

        dataset_train = tf.data.TFRecordDataset(train_records).map(parse_function)
        dataset_val = tf.data.TFRecordDataset(validation_records).map(parse_function)
        dataset_test = tf.data.TFRecordDataset(test_records).map(parse_function)

        # Calculate mean image, std image
        mean_image_path = os.path.join(LOCAL_FOLDER, 'mean_imagenet.npy')
        train_index_list_path = os.path.join(LOCAL_FOLDER, 'train_list.pkl')

        if not os.path.exists(mean_image_path):
            sess = tf.get_default_session()
            temp_iterator = dataset_train.batch(10).make_one_shot_iterator().get_next()
            p_mean = np.zeros(IMAGE_SHAPE)
            c = 0
            train_index_list=[]

            try:
                while True:
                    index, batch_x, batch_y = sess.run(temp_iterator)
                    p_mean = p_mean + np.mean(batch_x, axis=0)  # IMAGE_SHAPE
                    c += 1
                    train_index_list += index.tolist()
            except tf.errors.OutOfRangeError:
                self.mean = (p_mean / c).astype(np.float32)
                self.train_index_list = train_index_list
                np.save(mean_image_path, self.mean)
                with open(train_index_list_path,'wb') as f:
                    pickle.dump(train_index_list,f,-1)
                print("Saved mean image")
                print("Saved train index list")
        else:
            print("Mean image loaded from file")
            print("Train index list loaded from file")
            self.mean = np.load(mean_image_path)
            with open(train_index_list_path,'rb') as f:
                self.train_index_list = pickle.load(f)

        self.train_index_list = list(map(lambda x : x.decode('utf8'),self.train_index_list)) #each string was a byte array

        if add_data_augmentation:
            print('Using DATA AUGMENTATION')
            def preprocess(index, x, y):
                return (index, data_augmentation(tf.add(x, -self.mean) / 255,224), tf.one_hot(y, self.n_classes))
        else:
            def preprocess(index, x, y):
                return (index, tf.add(x, -self.mean) / 255, tf.one_hot(y, self.n_classes))

        # .apply(tf.contrib.data.map_and_batch( map_func=preprocess, batch_size=batch_size))
        # .cache()
        # .map(preprocess,num_parallel_calls=4).batch(batch_size)
        # .prefetch(1)
        # .shuffle(10)


        # preprocesss
        shuffle_buffer = int(self.n_classes*1000*0.6)
        self.train_dataset = dataset_train.map(preprocess, num_parallel_calls=3).shuffle(shuffle_buffer).cache().repeat(epochs).batch(
            batch_size).prefetch(3)
        self.valid_dataset = dataset_val.map(preprocess, num_parallel_calls=3).shuffle(shuffle_buffer).cache().repeat(1).batch(
            batch_size).prefetch(3)
        self.dataset_test = dataset_test.map(preprocess, num_parallel_calls=3).shuffle(shuffle_buffer).cache().repeat(1).batch(
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
        # example: n02423022_7746.JPEG
        # n02423022_original_images
        assert (hasattr(self, 'data_folder')), "Image folder undefined"
        if not(index in self.train_index_list):
            print("WARNING IMAGE NOT IN TRAIN LIST YOU ARE USING A TEST OR VALIDATION IMAGE {0}".format(index))
            if strict:
                print("Strict mode returning NONE")
                return None
        class_name = index.split('_')[0]
        full_path = os.path.join(self.data_folder, "{0}_original_images".format(class_name), index)
        img = cv2.imread(full_path)
        s = self.vis_shape()
        img_out = cv2.resize(img, tuple(s[0:2]))  # original image need resize

        img_out_rgb=cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB) # convert image from BGR to RGB

        return img_out_rgb.reshape(1, s[0], s[1], s[2]), [self.class_name_map[class_name]]

    def get_data_range(self):
        return [0, 255]

    def vis_shape(self):
        return IMAGE_SHAPE


if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        t = Imagenet_Dataset(1, 10,data_folder='./temp/imagenet_subset')
