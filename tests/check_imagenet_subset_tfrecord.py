import unittest

from datasets.imagenet_data import parse_function
from datasets.imagenet_data import list_records
from tf_records_parser.cifar10 import exist_tar, maybe_download_and_extract, cifar10_to_tfrecords, LOCAL_FOLDER
import tensorflow as tf
import cv2
import os
import numpy as np



class TestImagenetRecords:
    def __init__(self):
        tfrecords = list_records()
        assert(len(tfrecords) != 0),"No TF RECORDS FOUND FOR IMAGENET"

    def test_tfrecord_load(self):

        tfrecords = list_records()

        for filename in tfrecords:
            record_iterator = tf.python_io.tf_record_iterator(path=filename)

            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)

                height = int(example.features.feature['image/height']
                             .int64_list
                             .value[0])

                width = int(example.features.feature['image/width']
                            .int64_list
                            .value[0])
                depth = int(example.features.feature['image/channels']
                            .int64_list
                            .value[0])

                img_string = (example.features.feature['image/encoded']
                    .bytes_list
                    .value[0])


                img_1d = np.fromstring(img_string, dtype=np.uint8)
                reconstructed_img = cv2.imdecode(img_1d, -1)

                # import matplotlib.pyplot as plt
                # plt.imshow(reconstructed_img[:,:,::-1]) # Image is BGR to RGB
                # plt.show()
                expected_s = (width,height,depth)
                actual_s = reconstructed_img.shape

                if depth == 4 or depth == 3:
                    expected_s = (width, height, 3)
                    assert (expected_s == actual_s), "Expected shape: {0} Actual shape: {1}".format(expected_s,actual_s)
                elif depth == 1:
                    expected_s = (width, height)
                    assert (expected_s == actual_s), "Expected shape: {0} Actual shape: {1}".format(expected_s,actual_s)





    def test_tfrecord_dataset(self):

        tfrecords = list_records()

        dataset = tf.data.TFRecordDataset(tfrecords).map(parse_function).batch(1)

        batch= dataset.make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            bx,by=sess.run(batch)
            assert(bx.shape == (1,224,224,3))


if __name__ == '__main__':

    t=TestImagenetRecords()
    t.test_tfrecord_load()
    t.test_tfrecord_dataset()