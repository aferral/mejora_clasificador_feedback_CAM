from image_generator.abstract_generator import Abstract_generator, \
    try_to_adjust_to_3c_mask
import os
import tensorflow as tf
import neuralgym as ng
import sys
import numpy as np
path_m = os.path.abspath('../genm')
print(path_m)
sys.path.append(path_m)

from inpaint_model import InpaintCAModel



class yu2018generative(Abstract_generator):
    def __init__(self):
        self.generative_model_path = '../genm/model_logs/release_imagenet_256'

    def generate_img_mask(self,image,mask):

        mask,image = try_to_adjust_to_3c_mask(image, mask)

        assert (image.shape == mask.shape),"Mask shape {0} != img shape {1}".format(mask.shape,image.shape)


        # ng.get_gpus(1)

        model = InpaintCAModel()


        h, w, _ = image.shape
        grid = 8
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)



        temp_graph = tf.Graph()
        with temp_graph.as_default():
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True

            with tf.Session(config=sess_config) as sess:
                input_image = tf.constant(input_image, dtype=tf.float32)
                output = model.build_server_graph(input_image)
                output = (output + 1.) * 127.5
                output = tf.reverse(output, [-1])
                output = tf.saturate_cast(output, tf.uint8)
                # load pretrained model
                vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                assign_ops = []
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    var_value = tf.contrib.framework.load_variable(self.generative_model_path, from_name)
                    assign_ops.append(tf.assign(var, var_value))
                sess.run(assign_ops)
                result = sess.run(output)
                out_img = result[0][:, :, ::-1]
            return [out_img]

