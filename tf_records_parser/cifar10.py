import os
import tarfile
from six.moves import cPickle as pickle, urllib
import tensorflow as tf
import sys
import numpy as np

from utils import timeit

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'
LOCAL_FOLDER = "temp"

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def maybe_download_and_extract(data_dir = LOCAL_FOLDER,url = CIFAR_DOWNLOAD_URL):
  """Download and extract the tarball from Alex's website."""

  dest_directory = data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not exist_tar():
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
  file_names['validation'] = ['data_batch_5']
  file_names['eval'] = ['test_batch']
  return file_names


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    data_dict = pickle.load(f, encoding='latin1')
  return data_dict


def convert_to_tfrecord(input_file_list,output_file):
    writer = tf.python_io.TFRecordWriter(output_file)

    for file in input_file_list:
        numpy_data = read_pickle_from_file(file)
        for row in range(numpy_data['data'].shape[0]):
            img_flat = numpy_data['data'][row]
            label = numpy_data['labels'][row]
            height = 32
            width = 32
            depth = 3
            img_raw = img_flat.tostring()
            index = '{0} -- {1}'.format(os.path.split(file)[-1],row)
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'depth': _int64_feature(depth),
                'image_raw': _bytes_feature(img_raw),
                'label': _int64_feature(label),
                'index' : _bytes_feature(tf.compat.as_bytes(index))}))
            writer.write(example.SerializeToString())
    writer.close()

def exist_tar():
    filepath = os.path.join(LOCAL_FOLDER, CIFAR_FILENAME)
    return os.path.exists(filepath)

def cifar10_to_tfrecords():
    maybe_download_and_extract()
    file_names = _get_file_names()
    input_dir = os.path.join(LOCAL_FOLDER, CIFAR_LOCAL_FOLDER)
    for mode, files in file_names.items():
        print("Processing {0}".format(files))
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(LOCAL_FOLDER, mode + '.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(input_files, output_file)

    print('Done!')

if __name__ == '__main__':
    with timeit():
        cifar10_to_tfrecords()
