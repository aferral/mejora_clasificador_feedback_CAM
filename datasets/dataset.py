from sklearn.datasets import load_digits
import numpy as np
import tensorflow as tf
import os



class Dataset:
    def __init__(self):

        assert(hasattr(self, 'iterator')),"Dataset must define a iterator"

        assert (not (self.train_dataset is None)), "Dataset must define a train_dataset"
        assert (not (self.valid_dataset is None)), "Dataset must define a validation_dataset"

    def preprocess_batch(self,image_batch):
        raise NotImplementedError()

    def inverse_preprocess(self,image_batch):
        raise NotImplementedError()

    def get_iterator_entry(self):
        return self.iterator.get_next()

    def initialize_iterator_train(self,sess):
        training_init_op = self.iterator.make_initializer(self.train_dataset)
        sess.run(training_init_op)

    def initialize_iterator_val(self,sess):
        validation_init_op = self.iterator.make_initializer(self.valid_dataset)
        sess.run(validation_init_op)
        # validation_init_op.run()

    def get_data_range(self):
        raise NotImplementedError()

    def vis_shape(self):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()
    @property
    def shape_target(self):
        raise NotImplementedError()


    def get_train_image_at(self,index):
        """
        This function return an image in the trainset. Without the preprocessing step
        :param index:
        :return:
        """
        raise NotImplementedError()





class Digits_Dataset(Dataset):

    def __init__(self,epochs,batch_size):

        # Data load
        digits = load_digits(return_X_y=True)

        # split into train and validation sets
        self.train_images = train_images = digits[0][:int(len(digits[0]) * 0.8)]
        train_labels = digits[1][:int(len(digits[0]) * 0.8)]
        one_hot_train_labels = np.zeros((train_labels.shape[0], 10))
        one_hot_train_labels[np.arange(train_labels.shape[0]), train_labels] = 1

        valid_images = digits[0][int(len(digits[0]) * 0.8):]
        valid_labels = digits[1][int(len(digits[0]) * 0.8):]
        one_hot_val_labels = np.zeros((valid_labels.shape[0], 10))
        one_hot_val_labels[np.arange(valid_labels.shape[0]), valid_labels] = 1

        self.mean = train_images.mean(axis=0)

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
        iterator = self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                                   self.train_dataset.output_shapes)

        # check parameters
        super().__init__()


    def preprocess_batch(self,image_batch):
        """
        Process a new batch substract train_mean.
        :return:
        """
        if len(image_batch.shape) != 2 or image_batch.shape[1] != 64:
            image_batch = image_batch.reshape(-1,64)

        return image_batch - self.mean
    @property
    def shape(self):
        return [64]
    @property
    def shape_target(self):
        return [10]

    def vis_shape(self):
        return [8,8]

    def get_train_image_at(self, index):
        return self.train_images[index]

    def get_data_range(self):
        return [0,16]


if __name__ == "__main__":
    with tf.Session().as_default() as sess:
        a = Digits_Dataset(1,1)
        # a.get_train_image_at(0)