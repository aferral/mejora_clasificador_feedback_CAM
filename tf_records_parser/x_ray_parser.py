import os
import random

from skimage import io
import numpy as np
import tensorflow as tf

from tf_records_parser.cifar10 import _int64_feature, _bytes_feature
import cv2

RANDOM_SEED = 1
SAMPLES_PER_TFFILE = 500

def grayscaleEq(rgbimage):
    cof = 1.0/3
    return rgbimage[:,:,0] * cof + rgbimage[:,:,1] * cof + rgbimage[:,:,2] * cof


def write_to_tfrecord(numpy_array,label,index_str,writer):
    img_flat = numpy_array.flatten()
    height = 224
    width = 224
    depth = 1
    img_raw = img_flat.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'depth': _int64_feature(depth),
        'image_raw': _bytes_feature(img_raw),
        'label': _int64_feature(label),
        'index' : _bytes_feature(tf.compat.as_bytes(index_str))
    }))
    writer.write(example.SerializeToString())


def original_images_to_tfrecord(record_prefix,dataset_dir,folder_to_label,folder_to_name, output_dir,shuffling=False):

    # create out folder
    os.makedirs(output_dir,exist_ok=True)

    # open all file names and labels
    all_images_tuple = []
    folders = os.listdir(dataset_dir)

    ind_dict = {}
    c=0
    for folder in folders:
        path = os.path.join(dataset_dir, folder)
        if os.path.isdir(path):
            for image_name in os.listdir(path):
                full_path=os.path.join(path, image_name)
                index_str = "{0}_{1}".format(record_prefix,c)
                all_images_tuple.append((full_path,folder_to_label[folder], index_str))
                ind_dict[index_str] = full_path
                c+=1


    # shuffle if needed
    filenames = sorted(all_images_tuple)
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    N = len(filenames)
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = os.path.join(output_dir, "x_ray_{1}_{0}.tfrecord".format(fidx,record_prefix))
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_TFFILE:
                filename = filenames[i][0]
                label = filenames[i][1]
                index_str = filenames[i][2]
                data= cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2GRAY)
                write_to_tfrecord(data,label,index_str, tfrecord_writer)
                i += 1
                j += 1

                if (i % 100 == 0):
                    print("Image {0} of {1}".format(i,N))

            fidx += 1

    return ind_dict

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Calculate tf records from cwr images')
    parser.add_argument('folderIn', metavar='dataset_folder',
                        help='Where are the XRAY images')
    parser.add_argument('folderOut', metavar='out_folder',
                        help='Where will be the tf records stored')

    args = parser.parse_args()

    dataset_dir = args.folderIn
    output_dir = args.folderOut

    os.makedirs(output_dir,exist_ok=True)

    label_dict = {"Gun" : 0,"Other":1,"Razor":2,"Shuriken":3}
    folder_to_name = {x : x for x in label_dict}



    all_inds_dict = {}

    # Create train records
    train_imgs = os.path.join(dataset_dir,'Train')
    ind_dict=original_images_to_tfrecord("train", train_imgs, label_dict,folder_to_name, output_dir, shuffling=True)
    all_inds_dict = {**all_inds_dict,**ind_dict}


    # Create val records
    train_imgs = os.path.join(dataset_dir,'Valid')
    ind_dict=original_images_to_tfrecord("val", train_imgs, label_dict,folder_to_name, output_dir, shuffling=True)
    all_inds_dict = {**all_inds_dict, **ind_dict}


    # Create test records
    train_imgs = os.path.join(dataset_dir,'Test')
    ind_dict=original_images_to_tfrecord("test", train_imgs, label_dict,folder_to_name, output_dir, shuffling=True)
    all_inds_dict = {**all_inds_dict, **ind_dict}


    with open(os.path.join(output_dir,'labelnames.json'),'w') as f:
        full_data = {"names" : label_dict, "indexs" : all_inds_dict}
        json.dump(full_data,f)