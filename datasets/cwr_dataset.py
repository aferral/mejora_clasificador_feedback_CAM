import cv2
import tensorflow as tf
import os

from skimage import io

from datasets.dataset import Dataset
from tf_records_parser.cifar10 import LOCAL_FOLDER
import numpy as np

from tf_records_parser.cwr_parser import grayscaleEq


def parse_function(example_proto):
    features = {"image_raw": tf.FixedLenFeature((), tf.string),
                "label": tf.FixedLenFeature((), tf.int64),
                "height": tf.FixedLenFeature((), tf.int64),
                "width": tf.FixedLenFeature((), tf.int64),
                "depth": tf.FixedLenFeature((), tf.int64),
                "img_path": tf.FixedLenFeature((), tf.string)
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    h, w, d = parsed_features['height'], parsed_features['width'], parsed_features['depth']

    flat_image = tf.decode_raw(parsed_features["image_raw"], tf.float64)
    reconst = tf.cast(tf.transpose(tf.reshape(flat_image, tf.stack([d, h, w])), [1, 2, 0], 'reconstructed_image'),tf.float32)

    return parsed_features["img_path"],reconst, parsed_features["label"]

def get_records_folder():
    return  os.path.join(LOCAL_FOLDER, "CWR_records")

def list_records():
    path_records = get_records_folder()

    # list all tfrecords
    tfrecords = list(map(lambda x: os.path.join(path_records, x), sorted(filter(
        lambda name: name.split('.')[-1] == 'tfrecord',
        os.listdir(path_records)))))
    return tfrecords


class CWR_Dataset(Dataset):



    def __init__(self,epochs,batch_size,data_folder=None,**kwargs):

        if data_folder:
            self.data_folder = data_folder

        tfrecords = list_records()

        train_n_records = 18
        val_n_records = 3
        test_n_records = 3
        n_records = len(tfrecords)

        assert(train_n_records+val_n_records+test_n_records == n_records),"The train-val-test split must use all tf records."

        train_records = tfrecords[0:train_n_records]
        validation_records = tfrecords[train_n_records:train_n_records+val_n_records]
        test_records = tfrecords[train_n_records+val_n_records:]

        dataset_train = tf.data.TFRecordDataset(train_records).map(parse_function)
        dataset_val = tf.data.TFRecordDataset(validation_records).map(parse_function)
        dataset_test = tf.data.TFRecordDataset(test_records).map(parse_function)


        # Calculate mean image, std image
        mean_image_path = os.path.join(LOCAL_FOLDER,'mean_cwr.npy')

        if not os.path.exists(mean_image_path):
            sess = tf.get_default_session()
            temp_iterator = dataset_train.batch(300).make_one_shot_iterator().get_next()
            p_mean = np.zeros((96,96,1))
            c=0

            try:
                while True:
                    indexs,batch_x, batch_y = sess.run(temp_iterator)
                    p_mean = p_mean + np.mean(batch_x,axis=0) #(96,96,1)
                    c+=1
            except tf.errors.OutOfRangeError:
                self.mean = (p_mean / c).astype(np.float32)
                np.save(mean_image_path, self.mean)
                print("Saved mean image")
        else:
            print("Mean image loaded from file")
            self.mean = np.load(mean_image_path)

        def preprocess(indexs,x,y):
            return (indexs,tf.add(x, -self.mean) /255,tf.one_hot(y,4))

        # .apply(tf.contrib.data.map_and_batch( map_func=preprocess, batch_size=batch_size))
        # .cache()
        #.map(preprocess,num_parallel_calls=4).batch(batch_size)
        #.prefetch(1)
        #.shuffle(10)

        # preprocesss
        self.train_dataset = dataset_train.map(preprocess,num_parallel_calls=4).cache().repeat(epochs).batch(batch_size).prefetch(30)
        self.valid_dataset = dataset_val.map(preprocess,num_parallel_calls=4).cache().repeat(1).batch(batch_size).prefetch(30)
        self.dataset_test = dataset_test.map(preprocess,num_parallel_calls=4).cache().repeat(1).batch(batch_size).prefetch(30)


        # Create iterator
        iterator = self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                                   self.train_dataset.output_shapes)

        # check parameters
        super().__init__(**kwargs)

    def inverse_preprocess(self,image_batch):
        return image_batch*255 + self.mean

    def preprocess_batch(self,image_batch):
        """
        Process a new batch substract train_mean.
        :return:
        """
        if len(image_batch.shape) == 4 and image_batch.shape[1:] != (96,96,1):
            assert (False),'batch shape wrong'

        if len(image_batch.shape) == 3 and image_batch.shape != (96,96,1):
            assert(False),'batch shape wrong'

        return (image_batch - self.mean)/255

    @property
    def shape(self):
        return [96,96,1]
    @property
    def shape_target(self):
        return [4]

    def get_index_list(self):
        assert(hasattr(self,'data_folder')), "Image folder undefined"
        files=os.listdir(self.data_folder)
        return files


    def get_train_image_at(self, index):
        assert (hasattr(self, 'data_folder')), "Image folder undefined"
        img  = grayscaleEq(io.imread(os.path.join(self.data_folder, index)))
        label = index.split("_")[-1][0] # label given by image_label.jpg
        return img.reshape(1,96,96,1), label

    def get_data_range(self):
        return [0,255]

    def vis_shape(self):
        return [96,96]

if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        CWR_Dataset(1,10)