import tensorflow as tf
import matplotlib.pyplot as plt
from datasets.quickdraw_dataset import parse_function
from collections import Counter
IMAGE_SHAPE = [128, 128, 1]



with tf.Session() as sess:      
    for i in range(10,11):
        tr = '/home/aferral/PycharmProjects/generative_supervised_data_augmentation/temp/tf_records_quickdraw/sketchs_{0}.tfrecord'.format(i+1)
        dataset_train = tf.data.TFRecordDataset(tr).map(parse_function)

        temp_iterator = dataset_train.batch(10).make_one_shot_iterator().get_next()
        all_labels = []

        try:
            while True:
                index, batch_x, batch_y = sess.run(temp_iterator)
                all_labels += batch_y.tolist()

        except tf.errors.OutOfRangeError:
            print('For sketchs_{0}.tfrecord '.format(i+1))
            print('Counter : {0}'.format(Counter(all_labels)))
            print(" ")
            print(' ')
            print(" ")

