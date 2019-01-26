import pickle

import cv2
import tensorflow as tf
import os
from datasets.dataset import Dataset
from tf_records_parser.cifar10 import LOCAL_FOLDER
import numpy as np
import json
import re


IMAGE_SHAPE = [128, 128, 1]
folder_records = "tf_records_quickdraw"

def parse_function(example_proto):
    features = {"image_raw": tf.FixedLenFeature((), tf.string),
                "label": tf.FixedLenFeature((), tf.int64),
                "index": tf.FixedLenFeature((), tf.string)
                }
    parsed_features = tf.parse_single_example(example_proto, features)

    flat_image = tf.decode_raw(parsed_features["image_raw"], tf.uint8)
    reconst = tf.cast(tf.transpose(tf.reshape(flat_image, tf.stack([1, 128, 128])), [1, 2, 0], 'reconstructed_image'),tf.float32)

    return parsed_features["index"], reconst, parsed_features["label"]


def get_n_classes():
    class_map = os.path.join(LOCAL_FOLDER, folder_records, "labelnames.json")
    with open(class_map, 'r') as f:
        names_data = json.load(f)
    return len(names_data)
def get_records_folder():
    return os.path.join(LOCAL_FOLDER, folder_records)


def list_records():
    path_records = get_records_folder()

    # list all tfrecords
    tfrecords = list(map(lambda x: os.path.join(path_records, x), sorted(filter(
        lambda name: name.split('.')[-1] == 'tfrecord',
        os.listdir(path_records)))))
    return tfrecords


class QuickDraw_Dataset(Dataset):

    def __init__(self, epochs, batch_size, data_folder=None, **kwargs):
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
            self.class_name_map = {names_data[label].split('.')[0] : label for label in names_data}


        tfrecords.sort(key=lambda x: int(re.search('.+_(\d+).tfrecord', x).group(1)))
        train_records = [tfrecords[i] for i in [7,0,1,2,8,13,14,19,20]]
        validation_records = [tfrecords[i] for i in [5]] # ,3,4,5,11,15
        test_records = [tfrecords[i] for i in  [6]] # ,12,16
        #
        #,9,10,17,18

        #.cache('temp/dataset_cached_qd_tr.cache')
        #.cache('temp/dataset_cached_qd_val.cache')
        #.cache('temp/dataset_cached_qd_test.cache')

        train_n_records,val_n_records,test_n_records = len(train_records),len(validation_records),len(test_records)
        print("Train records : {0} , Val records: {1} , Test records {2}".format(train_n_records,len(validation_records),len(test_records)))
        #assert (train_n_records + val_n_records + test_n_records == n_records), "The train-val-test split must use all tf records."



        dataset_train = tf.data.TFRecordDataset(train_records).map(parse_function)
        dataset_val = tf.data.TFRecordDataset(validation_records).map(parse_function)
        dataset_test = tf.data.TFRecordDataset(test_records).map(parse_function)

        # Calculate mean image, std image
        mean_image_path = os.path.join(LOCAL_FOLDER, 'mean_quickdraw.npy')
        train_index_list_path = os.path.join(LOCAL_FOLDER, 'train_list_quickdraw.pkl')
        train_labels = os.path.join(LOCAL_FOLDER, 'labels_quickdraw.pkl')

        if not os.path.exists(mean_image_path):
            sess = tf.get_default_session()
            temp_iterator = dataset_train.batch(10).make_one_shot_iterator().get_next()
            p_mean = np.zeros(IMAGE_SHAPE)
            c = 0
            train_index_list=[]
            all_labels = []

            try:
                while True:
                    index, batch_x, batch_y = sess.run(temp_iterator)
                    p_mean = p_mean + np.mean(batch_x, axis=0)  # IMAGE_SHAPE
                    c += 1
                    train_index_list += index.tolist()
                    all_labels += batch_y.tolist()
            except tf.errors.OutOfRangeError:
                self.mean = (p_mean / c).astype(np.float32)
                self.train_index_list = train_index_list
                np.save(mean_image_path, self.mean)
                with open(train_index_list_path,'wb') as f:
                    pickle.dump(train_index_list,f,-1)



                with open(train_labels, 'wb') as f:
                    dict_labels = {ind : lbl for (ind,lbl) in zip(train_index_list,all_labels)}
                    pickle.dump(dict_labels, f, -1)
                print("Saved mean image")
                print("Saved train index list")
        else:
            print("Mean image loaded from file")
            print("Train index list loaded from file")
            self.mean = np.load(mean_image_path)
            with open(train_index_list_path,'rb') as f:
                self.train_index_list = pickle.load(f)

            with open(train_labels,'rb') as f:
                self.label_dict = pickle.load(f)
                self.label_dict = {k.decode('utf8') : self.label_dict[k] for k in self.label_dict}


        self.train_index_list = list(map(lambda x : x.decode('utf8'),self.train_index_list)) #each string was a byte array

        def preprocess(index, x, y):
            return (index, tf.add(x, -self.mean) / 255, tf.one_hot(y, self.n_classes))


        # preprocesss
        # shuffle_buffer = int(self.n_classes*1000*0.8)
        self.train_dataset = dataset_train.map(preprocess, num_parallel_calls=4).shuffle(29544).repeat(epochs).batch(batch_size)
        self.valid_dataset = dataset_val.map(preprocess, num_parallel_calls=4).shuffle(29544).repeat(1).batch(batch_size)
        self.dataset_test = dataset_test.map(preprocess, num_parallel_calls=4).shuffle(29544).repeat(1).batch(batch_size)

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
            print("WARNING IMAGE NOT IN TRAIN LIST YOU ARE USING A TEST OR VALIDATION IMAGE")
            if strict:
                print("Strict mode returning NONE")
                return None
        inverse_class_name = {int(self.class_name_map[k]) : k for k in self.class_name_map}
        class_name = inverse_class_name[self.label_dict[index]]
        full_path = os.path.join(self.data_folder,class_name, "{0}_{1}.png".format(class_name,index))
        img = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
        s = self.vis_shape()
        img_out = cv2.resize(img, tuple(s[0:2]))  # original image need resize

        return img_out.reshape(1, s[0], s[1], 1), [self.class_name_map[class_name]]

    def get_data_range(self):
        return [0, 255]

    def vis_shape(self):
        return [128,128]


if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        t = QuickDraw_Dataset(1, 10,data_folder='./temp/quickdraw_expanded_images')
