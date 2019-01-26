from classification_models.classification_model import CWR_classifier, \
    Abstract_model, digits_clasifier
from classification_models.custom_imagenet_subset import imagenet_classifier
from utils import imshow_util
from classification_models.vgg_16_batch_norm import vgg_16_batchnorm
from datasets.cifar10_data import Cifar10_Dataset
from datasets.cwr_dataset import CWR_Dataset
from datasets.dataset import Dataset, Digits_Dataset
from datasets.imagenet_data import Imagenet_Dataset
from datasets.quickdraw_dataset import QuickDraw_Dataset
from classification_models.imagenet_subset_cam_loss import imagenet_classifier_cam_loss

dataset_obj_dict = {
    'CWR' : CWR_Dataset,
    'Imagenet_subset' : Imagenet_Dataset,
    'cifar10' : Cifar10_Dataset,
    'digits' : Digits_Dataset,
    'quickdraw' : QuickDraw_Dataset
}

model_obj_dict = {
    'CWR_classifier': CWR_classifier,
    "vgg_16_batchnorm" : vgg_16_batchnorm,
    'digits' : digits_clasifier,
    "imagenet_custom" : imagenet_classifier,
    'imagenet_cam_loss' : imagenet_classifier_cam_loss
}

default_keys = ['mask_files','model_load_path','train_file_used']
default_values = [[],None,None]
requiered = [True,True,True]


# DEFAULT CONFIG DICTIONARY
m_config = {
    'mask_files' : [],
    'model_load_path' : None,
    'train_file_used' : None
}

