from sklearn.datasets import load_digits
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self):

        assert(hasattr(self, 'iterator')),"Dataset must define a iterator"

        # todo use this ??
        assert (not (self.train_dataset is None)), "Dataset must define a train_dataset"
        assert (not (self.valid_dataset is None)), "Dataset must define a validation_dataset"

    def preprocess_batch(self,image_batch):
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

    @property
    def shape(self):
        raise NotImplementedError()
    @property
    def shape_target(self):
        raise NotImplementedError()


    # todo use this?
    def get_batch(self):
        raise NotImplementedError()





class Digits_Dataset(Dataset):

    def __init__(self,epochs,batch_size):


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
        return image_batch - self.mean
    @property
    def shape(self):
        return [64]
    @property
    def shape_target(self):
        return [10]




if __name__ == "__main__":
    a = Digits_Dataset(1,1)