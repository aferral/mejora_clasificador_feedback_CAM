import shutil
from ImageNet_Utils.libs.imagedownloader import ImageNetDownloader
import os
import json
import random
import tensorflow as tf
from shutil import copyfile
from PIL import Image
from skimage.io import imread
#Based in https://github.com/balancap/SSD-Tensorflow/blob/master/datasets/pascalvoc_to_tfrecords.py


config_path = 'imagenet_config.json'
if not os.path.exists(config_path):
    with open(config_path,'w') as f:
        json.dump({"username" : "HERE_USERNAME","accessKey" : "Go to image-net.org/download-images if you dont have one"},f)
    raise Exception("CONFIGURE FIRST imagenet_config.json in {0}".format(os.path.abspath(config_path)))
else:
    print("Trying to load imagenet_config. At {0}".format(os.path.abspath(config_path)))
with open(config_path,'r') as f:
    config_dict = json.load(f)
    username = config_dict['username']
    accessKey = config_dict['accessKey']

    assert(not (username == "HERE_USERNAME"))," Edit the config_file"
    assert(not (accessKey == "Go to image-net.org/download-images if you dont have one"))," Edit the config_file"

BASE_FOLDER = os.getcwd()
RANDOM_SEED = 1
SAMPLES_PER_TFFILE = 1000

downloader = ImageNetDownloader()

# class_name : (label_index, imagenet_index)  class_name-label_index can be anything imagenet_index should be valid
subset = {
"Gazella" : [0,"n02423022"],
"Sea lion" : [1,"n02077923"],
"Pandas" : [2,"n02509815"],
"Elefante" : [3,"n02504458"] ,
"Gato tigre" : [4,"n02123159"],
"Coyote" : [5,"n02114855"],

"Zebra" : [6,"n02391049"],
"Gorila" : [7,"n02480855"],
"Camello" : [8,"n02437312"],
"Artic fox" : [9,"n02120079"],
"Mono aullador" : [10,"n02492660"],
"Armadillo" : [11,"n02454379"],
"Red wolf" : [12,"n02114712"],
"Llama" : [13,"n02437616"],

"Otter" : [14,"n02444819"],
"Toucan" : [15,"n01843383"],
"Pavoreal" : [16,"n01806143"],
"Buo gris" : [17,"n01622779"],
"Aguila blanca" : [18,"n01614925"],
"Manta raya" : [19,"n01498041"],
"Tiburon martillo" : [20,"n01494475"],
}


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _add_to_tfrecord(filename, label_name,label_index, tfrecord_writer):

    # Read the image file.
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    im = Image.open(filename)

    shape = im.size

    # todo resize????

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(im.layers),
            'image/shape': int64_feature(list(shape)),
            'image/label': int64_feature(label_index),
            'image/label_text': bytes_feature(label_name.encode('ascii')),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))

    tfrecord_writer.write(example.SerializeToString())


def get_image_folders():
    folder_list=os.listdir(BASE_FOLDER)
    return list(filter(lambda name : name[0] == 'n',folder_list))


def download_subset(out_folder):
    # # Download images
    for values in subset.values():
        id_imagenet = values[1]
        downloader.downloadOriginalImages(id_imagenet, username, accessKey)

    # select n* folders
    folders = get_image_folders()
    print("{1} folders downloaded: {0}".format(folders,len(folders)))

    # Create folder in temp
    os.makedirs(out_folder,exist_ok=True)

    # Delete .tar, move folders and delete download folders
    for elem in folders:
        for content in os.listdir(os.path.join(BASE_FOLDER,elem)):
            path = os.path.join(BASE_FOLDER,elem,content)
            if os.path.isdir(path) and ('original_images' in content):
                shutil.move(path, os.path.join(out_folder,content))
            if content.split('.')[-1] == 'tar':
                print("Deleting {0}".format(content))
                os.remove(path)
        shutil.rmtree(os.path.join(BASE_FOLDER,elem))

    # Save label names
    with open(os.path.join(out_folder,'labelnames.json'), 'w') as outfile:
        json.dump(subset, outfile)

def original_images_to_tfrecord(dataset_dir, output_dir,shuffling=False):

    # create out folder
    os.makedirs(output_dir,exist_ok=True)

    # open label dict
    with open(os.path.join(dataset_dir,'labelnames.json')) as f:
        d = json.load(f)
    label_dict = {key: ind for ind, key in d.values()} # ex:  'n02437312': 8
    name_dict = {d[name][1]: name for name in d}  # ex:  'n02437312': 'Gorilla'

    # Copy labels json for future reference
    copyfile(os.path.join(dataset_dir,'labelnames.json'), os.path.join(output_dir,'labelnames.json'))

    # open all file names and labels
    all_images_tuple = []
    folders = os.listdir(dataset_dir)
    for folder in folders:
        path = os.path.join(dataset_dir, folder)
        if os.path.isdir(path):
            for image_name in os.listdir(path):
                imagenet_index = image_name.split('_')[0]
                all_images_tuple.append((os.path.join(path, image_name),label_dict[imagenet_index], name_dict[imagenet_index]))


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
        tf_filename = os.path.join(output_dir, "imagenet_subset_{0}.tfrecord".format(fidx))
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_TFFILE:
                filename = filenames[i][0]
                label = filenames[i][1]
                label_name = filenames[i][2]
                _add_to_tfrecord(filename, label_name,label, tfrecord_writer)
                i += 1
                j += 1

                if (i % 100 == 0):
                    print("Image {0} of {1}".format(i,N))

            fidx += 1


if __name__ == '__main__':

    out_folder = './temp/imagenet_subset'
    out_tfrecods = './temp/tfrecords_imagenet_subset/'


    print("This function will download a IMAGENET subset (may take a lot of time). ")
    print("The subset: ")
    for elem in subset:
        print("k: {0} v: {1}".format(elem,subset[elem]))
    print("-------------")
    print("-------------")
    input("Press ENTER to continue")


    print("Downloading images from IMAGENET")
    # download_subset(out_folder)
    print("Download ended. Outfolder: {0}".format(out_folder))


    print("Starting tfrecord conversion")
    original_images_to_tfrecord(out_folder, out_tfrecods, shuffling=True)
    print("Tfrecord conversion ended. Output at {0}".format(out_tfrecods))