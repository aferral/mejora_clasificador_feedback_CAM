from datasets.dataset import Dataset
from datasets.imagenet_data import Imagenet_Dataset
from image_generator.abstract_generator import Abstract_generator, try_to_adjust_to_shape_mask
import random
import numpy as np

class Replace_with_dataset_crops(Abstract_generator):

    def generate_img_mask(self, img, mask):

        mask, img = try_to_adjust_to_shape_mask(img, mask)
        mask = mask.astype(np.bool)
        assert (img.shape == mask.shape),"Mask shape {0} != img shape {1}".format(mask.shape,img.shape)


        out_images = []

        for i in range(self.gen_per_image):
            random_index = random.choice(self.index_list)
            img_dataset,label = self.dataset.get_train_image_at(random_index)

            # use the mask to replace the img with the random index image
            random_replace =  img.copy()

            random_replace[mask] = img_dataset[0][mask]

            out_images.append(random_replace)

        return out_images



    def __init__(self,dataset : Dataset,gen_per_image):
        self.dataset = dataset
        self.index_list = self.dataset.get_index_list()
        self.gen_per_image = gen_per_image
        pass

if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    with tf.Session().as_default() as sess:
        dataset_obj = Imagenet_Dataset(1, 10,data_folder='./temp/imagenet_subset')
        gen=Replace_with_dataset_crops(dataset_obj,3)

        mask = np.zeros((224,224))
        mask[100:200,100:200] = 1
        mask=mask.astype(np.bool)

        img_org,label = dataset_obj.get_train_image_at("n02114712_171.JPEG")
        img_org = img_org[0]
        plt.imshow(img_org)
        plt.show()

        generated = gen.generate_img_mask(img_org,mask)
        for x in generated:
            plt.imshow(x)
            plt.show()

    pass
