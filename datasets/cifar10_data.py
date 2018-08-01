import tensorflow as tf
import os
from datasets.dataset import Dataset
from tf_records_parser.cifar10 import LOCAL_FOLDER, read_pickle_from_file
import numpy as np

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


class Cifar10_Dataset(Dataset):


    def __init__(self,epochs,batch_size,data_folder=None,**kwargs):


        if data_folder:
            self.data_folder = data_folder

        train_records = os.path.join(LOCAL_FOLDER,'train.tfrecords')
        validation_records = os.path.join(LOCAL_FOLDER, 'validation.tfrecords')
        test_records = os.path.join(LOCAL_FOLDER, 'eval.tfrecords')

        dataset_train = tf.data.TFRecordDataset(train_records).map(parse_function)
        dataset_val = tf.data.TFRecordDataset(validation_records).map(parse_function)
        dataset_test = tf.data.TFRecordDataset(test_records).map(parse_function)


        # Calculate mean image, std image
        mean_image_path = os.path.join(LOCAL_FOLDER,'mean.npy')

        if not os.path.exists(mean_image_path):
            sess = tf.get_default_session()
            temp_iterator = dataset_train.batch(300).make_one_shot_iterator().get_next()
            p_mean = np.zeros((32,32,3))
            c=0

            try:
                while True:
                    index,batch_x, batch_y = sess.run(temp_iterator)
                    p_mean = p_mean + np.mean(batch_x,axis=0) #(32,32,3)
                    c+=1
            except tf.errors.OutOfRangeError:
                self.mean = (p_mean / c).astype(np.float32)
                np.save(mean_image_path, self.mean)
                print("Saved mean image")
        else:
            print("Mean image loaded from file")
            self.mean = np.load(mean_image_path)

        def preprocess(index,x,y):
            return (index,tf.add(x, -self.mean),tf.one_hot(y,10))

        # preprocesss
        self.train_dataset = dataset_train.map(preprocess).shuffle(500).repeat(epochs).batch(batch_size).cache().prefetch(2000)
        self.valid_dataset = dataset_val.map(preprocess).shuffle(500).repeat(1).batch(batch_size).cache().prefetch(2000)
        self.dataset_test = dataset_test.map(preprocess).shuffle(500).repeat(1).batch(batch_size).cache().prefetch(2000)


        # Create iterator
        iterator = self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                                   self.train_dataset.output_shapes)

        # check parameters
        super().__init__(**kwargs)

    def inverse_preprocess(self,image_batch):
        return image_batch + self.mean

    def preprocess_batch(self,image_batch):
        """
        Process a new batch substract train_mean.
        :return:
        """
        if len(image_batch.shape) == 4 and image_batch.shape[1:] != (32,32,3):
            assert (False),'batch shape wrong'

        if len(image_batch.shape) == 3 and image_batch.shape != (32,32,3):
            assert(False),'batch shape wrong'

        return image_batch - self.mean

    @property
    def shape(self):
        return [32,32,3]
    @property
    def shape_target(self):
        return [10]

    def get_index_list(self):
        assert(hasattr(self,'data_folder')), "Image folder undefined"

        data_f = ['data_batch_{0}'.format(i) for i in range(1,6)]
        data_f += ['test_batch']

        indexs = []
        for f in data_f:
            for j in range(10000):
                indexs.append('{0} -- {1}'.format(f,j))

        return indexs


    def get_train_image_at(self, index):
        # example: data_batch_1 -- 0
        assert (hasattr(self, 'data_folder')), "Image folder undefined"

        if type(index) == str:
            file,row=index.split('--')
        else:
            file,row=index.decode('utf-8').split('--')
        file=file.strip()
        row=int(row.strip())

        full_path = os.path.join(self.data_folder, file)
        data=read_pickle_from_file(full_path)
        img=data['data'][row]
        label=data['labels'][row]
        s=self.vis_shape()
        img_out = np.swapaxes(img.reshape(3,32,32),0,2)
        return img_out.reshape(1,s[0],s[1],s[2]), [label]


    def get_data_range(self):
        return [0,255]

    def vis_shape(self):
        return [32,32,3]

if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        Cifar10_Dataset(1,10)