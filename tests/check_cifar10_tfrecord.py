import unittest

from datasets.cifar10_data import parse_function
from tf_records_parser.cifar10 import exist_tar, maybe_download_and_extract, cifar10_to_tfrecords, LOCAL_FOLDER
import tensorflow as tf

import os
import numpy as np


def list_records(folder):
    only_records = filter(lambda fp: ".tfrecords" in fp, os.listdir(folder))
    full_path = map(lambda fp : os.path.join(folder,fp),only_records)
    return full_path


class TestCifar10Records:
    def __init__(self):
        if not exist_tar():
            maybe_download_and_extract()
            cifar10_to_tfrecords()
    def test_tfrecord_load(self):
        for filename in list_records(LOCAL_FOLDER):
            record_iterator = tf.python_io.tf_record_iterator(path=filename)

            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)

                height = int(example.features.feature['height']
                             .int64_list
                             .value[0])

                width = int(example.features.feature['width']
                            .int64_list
                            .value[0])
                depth = int(example.features.feature['depth']
                            .int64_list
                            .value[0])

                img_string = (example.features.feature['image_raw']
                    .bytes_list
                    .value[0])

                label = (example.features.feature['label']
                    .int64_list
                    .value[0])

                img_1d = np.fromstring(img_string, dtype=np.uint8)
                reconstructed_img = np.transpose(img_1d.reshape((depth,height, width)),[1,2,0])

                assert(reconstructed_img.shape == (32,32,3)),'Error reconstructed image shape != (32,32,3)'
    def test_tfrecord_dataset(self):

        list_tf_records = list(list_records(LOCAL_FOLDER))

        dataset = tf.data.TFRecordDataset(list_tf_records).map(parse_function).batch(10)

        batch= dataset.make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            bx,by=sess.run(batch)
            assert(bx.shape == (10,32,32,3))


if __name__ == '__main__':

    t=TestCifar10Records()
    t.test_tfrecord_load()
    t.test_tfrecord_dataset()