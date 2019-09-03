import cv2
import tensorflow as tf
import os

from skimage import io

from datasets.dataset import Dataset
from tf_records_parser.cifar10 import LOCAL_FOLDER
import numpy as np
import cv2

from tf_records_parser.cwr_parser import grayscaleEq





def read_img(path):
    temp = cv2.bitwise_not(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    return np.expand_dims(temp.astype(np.float32),axis=-1)

class Simple_figures_dataset(Dataset):



    def __init__(self,epochs,batch_size,data_folder = './temp/dataset_simbols',**kwargs):

        self.class_labels = {'xi' : 0, 'epsilon' : 1, 'zeta' : 2}
        self.r_class_labels = {self.class_labels[k] : k for k in self.class_labels}

        self.all_indexs = []
        self.data_folder = data_folder



        data_train = []
        data_test = []

        def process_folder(fname,name):
            folder_path = os.path.join(self.data_folder, fname)
            label = self.class_labels[name]
            files = os.listdir(folder_path)

            map_f = lambda x : (x,read_img(os.path.join(folder_path, x)),label)

            return list(map(map_f, files ) )

        train_folders = [(name, 'out_{0}'.format(name)) for name in
                         self.class_labels]
        for name,fname in train_folders:
            data_train += process_folder(fname,name)


        test_folders = [(name, 'out_test_{0}'.format(name)) for name in self.class_labels]
        for name,fname in test_folders:
            data_test += process_folder(fname,name)


        # pasar a tf record
        def slice_col_list(list_t,col):
            return list(map(lambda x : x[col],list_t))

        self.all_indexs += slice_col_list(data_train,0)

        inds = tf.data.Dataset.from_tensor_slices(np.array(slice_col_list(data_train, 0)))
        imgs = tf.data.Dataset.from_tensor_slices(np.array(slice_col_list(data_train,1)))
        labels = tf.data.Dataset.from_tensor_slices(np.array(slice_col_list(data_train, 2)))
        dataset_train = tf.data.Dataset.zip((inds,imgs,labels))


        dataset_val = dataset_train



        inds = tf.data.Dataset.from_tensor_slices(np.array(slice_col_list(data_test, 0)))
        imgs = tf.data.Dataset.from_tensor_slices(np.array(slice_col_list(data_test,1)))
        labels = tf.data.Dataset.from_tensor_slices(np.array(slice_col_list(data_test, 2)))
        dataset_test = tf.data.Dataset.zip((inds,imgs,labels))

        self.mean = np.zeros_like(data_train[0][1])

        def preprocess(indexs,x,y):
            return (indexs,x /255,tf.one_hot(y,3))

        # preprocesss
        self.train_dataset = dataset_train.map(preprocess,num_parallel_calls=4).shuffle(300).repeat(epochs).batch(batch_size)
        self.valid_dataset = dataset_val.map(preprocess,num_parallel_calls=4).shuffle(300).batch(batch_size)
        self.dataset_test = dataset_test.map(preprocess,num_parallel_calls=4).shuffle(300).batch(batch_size)


        #self.dataset_test, self.train_dataset = self.train_dataset,self.dataset_test
        #self.train_dataset = self.dataset_test


        # Create iterator
        iterator = self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                                   self.train_dataset.output_shapes)

        # check parameters
        super().__init__(**kwargs)

    def inverse_preprocess(self,image_batch):
        return image_batch*255

    def preprocess_batch(self,image_batch):
        """
        Process a new batch substract train_mean.
        :return:
        """
        if len(image_batch.shape) == 4 and image_batch.shape[1:] != (72,72,1):
            assert (False),'wrong batch shape'

        if len(image_batch.shape) == 3:
            if image_batch.shape == (72, 72, 3):
                image_batch = np.mean(image_batch,axis=2).reshape(72,72,1)
            elif image_batch.shape == (72, 72, 1):
                pass
            else:
                assert(False),'wrong batch shape'

        return image_batch/255

    @property
    def shape(self):
        return [72,72,1]
    @property
    def shape_target(self):
        return [3]

    def get_index_list(self):
        return self.all_indexs


    def get_train_image_at(self, index):

        if 'test' in index:
            print('WARNING ACCESING TEST IMAGE')

        parts = index.split('_')
        folder = 'out_{0}'.format('_'.join(parts[:-1]))
        label = self.class_labels[parts[-2]]

        return np.expand_dims(read_img(os.path.join(self.data_folder,folder,index)),axis=0), np.expand_dims(label,axis=0)


    def get_data_range(self):
        return [0,255]

    def vis_shape(self):
        return [72,72]

if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        t2=Simple_figures_dataset(1,10)
        print(t2.get_index_list())
        print(t2.get_train_image_at('test_zeta_3.png'))
        print(t2.get_train_image_at('zeta_3.png'))
