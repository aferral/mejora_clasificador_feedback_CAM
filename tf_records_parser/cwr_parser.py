import os
import random

from skimage import io
import numpy as np
import tensorflow as tf

from tf_records_parser.cifar10 import _int64_feature, _bytes_feature


RANDOM_SEED = 1
SAMPLES_PER_TFFILE = 500

def grayscaleEq(rgbimage):
    cof = 1.0/3
    return rgbimage[:,:,0] * cof + rgbimage[:,:,1] * cof + rgbimage[:,:,2] * cof


def write_to_tfrecord(numpy_array,label,writer):
    img_flat = numpy_array.flatten()
    height = 96
    width = 96
    depth = 1
    img_raw = img_flat.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'depth': _int64_feature(depth),
        'image_raw': _bytes_feature(img_raw),
        'label': _int64_feature(label)}))
    writer.write(example.SerializeToString())

def cwr_to_tfrecord(dataFolder,output_dir,shuffling=True):

    os.makedirs(output_dir,exist_ok=True)

    # Here open files in folder somehow
    all = []
    allL = []
    fileList = os.listdir(dataFolder)

    # shuffle if needed
    if shuffling:
        fileList = sorted(fileList)
        random.seed(RANDOM_SEED)
        random.shuffle(fileList)


    supImage = ["jpg", 'png']
    c=0
    # Read all the images and labels
    for ind, f in enumerate(fileList):
        if f.split('.')[-1] in supImage:
            gray_image = grayscaleEq(io.imread(os.path.join(dataFolder, f)))
            label = int(f.split("_")[1].split('.')[0])
            all.append(gray_image)
            c+=1
            allL.append(label)




    # Process dataset files.
    i = 0
    fidx = 0
    N = len(all)
    while i < len(all):
        # Open new TFRecord file.
        tf_filename = os.path.join(output_dir, "CWR_{0}.tfrecord".format(fidx))
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(all) and j < SAMPLES_PER_TFFILE:
                gray_img = all[i]
                label = allL[i]
                write_to_tfrecord(gray_img, label, tfrecord_writer)
                i += 1
                j += 1

                if (i % 100 == 0):
                    print("Image {0} of {1}".format(i,N))

            fidx += 1


if __name__ == "__main__":
    cwr_to_tfrecord("./temp/CW96Scalograms",'./temp/CWR_records/')