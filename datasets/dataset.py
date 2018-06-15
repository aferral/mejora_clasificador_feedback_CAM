from abc import abstractmethod

from sklearn.datasets import load_digits
import numpy as np
import tensorflow as tf
import os



class Dataset:
    def __init__(self,test_mode=False):

        self.test_mode=test_mode # All datasets return a single batch and end

        assert(hasattr(self, 'iterator')),"Dataset must define a iterator"

        assert (not (self.train_dataset is None)), "Dataset must define a train_dataset"
        assert (not (self.valid_dataset is None)), "Dataset must define a validation_dataset"
        assert (not (self.dataset_test is None)), "Dataset must define a test_dataset)"

    @abstractmethod
    def preprocess_batch(self,image_batch):
        raise NotImplementedError()

    @abstractmethod
    def inverse_preprocess(self,image_batch):
        raise NotImplementedError()

    def get_iterator_entry(self):
        return self.iterator.get_next()

    def _prepare_iterator(self,sess,dataset):
        if self.test_mode:
            sess.run(self.iterator.make_initializer(dataset.take(1)))
        else:
            sess.run(self.iterator.make_initializer(dataset))

    def initialize_iterator_train(self,sess):
        self._prepare_iterator(sess,self.train_dataset)

    def initialize_iterator_val(self,sess):
        self._prepare_iterator(sess,self.valid_dataset)

    @abstractmethod
    def get_data_range(self):
        raise NotImplementedError()

    @abstractmethod
    def vis_shape(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError()
    @property
    @abstractmethod
    def shape_target(self):
        raise NotImplementedError()

    @abstractmethod
    def get_train_image_at(self,index):
        """
        This function return an image in the trainset. Without the preprocessing step
        :param index:
        :return:
        """
        raise NotImplementedError()





class Digits_Dataset(Dataset):

    def __init__(self,epochs,batch_size,**kargs):

        # Data load
        digits = load_digits(return_X_y=True)

        self.all_data_x =digits[0]
        self.all_data_y = digits[1]
        train_st = int(len(digits[0]) * 0.8)

        # split into train and validation sets
        self.train_images = train_images = self.all_data_x[:train_st]
        train_labels = self.all_data_y[:train_st]
        one_hot_train_labels = np.zeros((train_labels.shape[0], 10))
        one_hot_train_labels[np.arange(train_labels.shape[0]), train_labels] = 1

        valid_images = self.all_data_x[train_st:]
        valid_labels = self.all_data_y[train_st:]
        one_hot_val_labels = np.zeros((valid_labels.shape[0], 10))
        one_hot_val_labels[np.arange(valid_labels.shape[0]), valid_labels] = 1

        all_indexs =  np.arange(self.all_data_x.shape[0])
        indexs_train = all_indexs[:train_st]
        indexs_val = all_indexs[train_st:]

        self.mean = train_images.mean(axis=0)

        # Create dataset objects
        dx_train = tf.data.Dataset.from_tensor_slices(train_images).map(lambda z: tf.add(z, -self.mean))
        dy_train = tf.data.Dataset.from_tensor_slices(one_hot_train_labels)
        indx_t_dataset = tf.data.Dataset.from_tensor_slices(indexs_train)
        self.train_dataset = tf.data.Dataset.zip((indx_t_dataset,dx_train, dy_train)).shuffle(500).repeat(epochs).batch(
            batch_size).cache().prefetch(2000)

        dx_valid = tf.data.Dataset.from_tensor_slices(valid_images).map(lambda z: tf.add(z, -self.mean))
        dy_valid = tf.data.Dataset.from_tensor_slices(one_hot_val_labels)
        indx_v_dataset = tf.data.Dataset.from_tensor_slices(indexs_val)
        self.dataset_test = self.valid_dataset = tf.data.Dataset.zip((indx_v_dataset,dx_valid, dy_valid)).shuffle(500).repeat(1).batch(
            batch_size).cache().prefetch(2000)

        # Create iterator
        iterator = self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                                   self.train_dataset.output_shapes)

        # check parameters
        super().__init__(**kargs)


    def preprocess_batch(self,image_batch):
        """
        Process a new batch substract train_mean.
        :return:
        """
        if len(image_batch.shape) != 2 or image_batch.shape[1] != 64:
            image_batch = image_batch.reshape(-1,64)

        return image_batch - self.mean

    def inverse_preprocess(self,image_batch):
        if len(image_batch.shape) != 2 or image_batch.shape[1] != 64:
            image_batch = image_batch.reshape(-1,64)

        return image_batch + self.mean



    @property
    def shape(self):
        return [64]
    @property
    def shape_target(self):
        return [10]

    def vis_shape(self):
        return [8,8]

    def get_train_image_at(self, index):
        return self.all_data_x[index]

    def get_data_range(self):
        return [0,16]


if __name__ == "__main__":
    with tf.Session().as_default() as sess:
        a = Digits_Dataset(1,1)
        # a.get_train_image_at(0)