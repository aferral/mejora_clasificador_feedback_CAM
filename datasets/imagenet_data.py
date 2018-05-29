import tensorflow as tf
import os
from datasets.dataset import Dataset
from tf_records_parser.cifar10 import LOCAL_FOLDER
import numpy as np

IMAGE_SHAPE = [224,224,3]

def parse_function(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string),
                "image/label": tf.FixedLenFeature((), tf.int64),
                "image/height": tf.FixedLenFeature((), tf.int64),
                "image/width": tf.FixedLenFeature((), tf.int64),
                "image/channels": tf.FixedLenFeature((), tf.int64)
                }
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_jpeg(parsed_features["image/encoded"],channels=3)

    # tf.Assert(tf.shape(image)[2] == 3,[image])
    reconst = tf.image.resize_images(image,IMAGE_SHAPE[0:2])

    reconst = tf.cast(reconst,tf.float32,name='image_reshaped')

    return reconst, parsed_features["image/label"]

def get_records_folder():
    return  os.path.join(LOCAL_FOLDER, "tfrecords_imagenet_subset")

def list_records():
    path_records = get_records_folder()

    # list all tfrecords
    tfrecords = list(map(lambda x: os.path.join(path_records, x), sorted(filter(
        lambda name: name.split('.')[-1] == 'tfrecord',
        os.listdir(path_records)))))
    return tfrecords

class Imagenet_Dataset(Dataset):

    def __init__(self,epochs,batch_size):
        tfrecords = list_records()

        train_n_records = 26
        val_n_records = 2
        test_n_records = 2
        n_records = len(tfrecords)

        assert(train_n_records+val_n_records+test_n_records == n_records),"The train-val-test split must use all tf records."

        train_records = tfrecords[0:train_n_records]
        validation_records = tfrecords[train_n_records:train_n_records+val_n_records]
        test_records = tfrecords[train_n_records+val_n_records:]

        dataset_train = tf.data.TFRecordDataset(train_records).map(parse_function)
        dataset_val = tf.data.TFRecordDataset(validation_records).map(parse_function)
        dataset_test = tf.data.TFRecordDataset(test_records).map(parse_function)


        # Calculate mean image, std image
        mean_image_path = os.path.join(LOCAL_FOLDER,'mean_imagenet.npy')

        if not os.path.exists(mean_image_path):
            sess = tf.get_default_session()
            temp_iterator = dataset_train.batch(10).make_one_shot_iterator().get_next()
            p_mean = np.zeros(IMAGE_SHAPE)
            c=0

            try:
                while True:
                    batch_x, batch_y = sess.run(temp_iterator)
                    p_mean = p_mean + np.mean(batch_x,axis=0) # IMAGE_SHAPE
                    c+=1
            except tf.errors.OutOfRangeError:
                self.mean = (p_mean / c).astype(np.float32)
                np.save(mean_image_path, self.mean)
                print("Saved mean image")
        else:
            print("Mean image loaded from file")
            self.mean = np.load(mean_image_path)

        def preprocess(x,y):
            return (tf.add(x, -self.mean) /255,tf.one_hot(y,21))

        # .apply(tf.contrib.data.map_and_batch( map_func=preprocess, batch_size=batch_size))
        # .cache()
        #.map(preprocess,num_parallel_calls=4).batch(batch_size)
        #.prefetch(1)
        #.shuffle(10)

        # preprocesss
        self.train_dataset = dataset_train.map(preprocess,num_parallel_calls=4).cache().repeat(epochs).batch(batch_size).prefetch(3)
        self.valid_dataset = dataset_val.map(preprocess,num_parallel_calls=4).cache().repeat(1).batch(batch_size).prefetch(3)
        self.dataset_test = dataset_test.map(preprocess,num_parallel_calls=4).cache().repeat(1).batch(batch_size).prefetch(3)


        # Create iterator
        iterator = self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                                   self.train_dataset.output_shapes)

        # check parameters
        super().__init__()

    def preprocess_batch(self,image_batch):
        """
        Process a new batch substract train_mean.
        :return:
        """

        return (image_batch - self.mean) /255

    @property
    def shape(self):
        return IMAGE_SHAPE
    @property
    def shape_target(self):
        return [21]

    def get_train_image_at(self, index):
        sess = tf.get_default_session()
        temp_iterator = self.train_dataset.make_one_shot_iterator().get_next()
        batch_x, batch_y = sess.run(temp_iterator)
        return (batch_x * 255) + self.mean

    def get_data_range(self):
        return [0,255]

    def vis_shape(self):
        return IMAGE_SHAPE

if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        Imagenet_Dataset(1,10)