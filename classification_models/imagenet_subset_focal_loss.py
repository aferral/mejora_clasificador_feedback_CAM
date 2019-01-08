


"""


"""
from classification_models.imagenet_subset_cam_loss import imagenet_classifier_cam_loss, get_slim_arch_bn
import tensorflow as tf

class imagenet_classifier_focal_loss(imagenet_classifier_cam_loss):

    def define_arch(self):
        phase = tf.placeholder(tf.bool, name='phase')

        # Define the model:
        predictions, acts = get_slim_arch_bn(self.input_l, phase,
                                             self.dataset.shape_target[0])

        # Configure values for visualization

        self.last_conv = acts['vgg_16/conv5/conv5_3']
        self.softmax_weights = r"vgg_16/softmax_logits/weights:0"
        self.pred = tf.nn.softmax(predictions, name='prediction')
        self.cam_out = self.graph.get_tensor_by_name('vgg_16/raw_CAM/cam_out:0')

        with tf.variable_scope("cam_loss_term"):
            sq_cam = tf.squeeze(self.cam_out, axis=[3, 4])  # N x hl x wl x C

            # selecciono solamente CAM del target N x hl x wl x 1
            sel_index = tf.cast(tf.argmax(self.targets, axis=1), tf.int32)
            sel_index = tf.stack([tf.range(tf.shape(sq_cam)[0]), sel_index],
                                 axis=1, name='selected_index')

            # esto es algo complejo pero lo unico que hace es seleccionar por canal el del indice
            selected_cam = tf.gather_nd(tf.transpose(sq_cam, perm=[0, 3, 1, 2]),
                                        sel_index, name='selected_cam')

            # nuevo termino de loss = CAM[label](upsample)[mask].sum() * ponderador
            # placeholders
            cam_mask = tf.placeholder_with_default(tf.zeros_like(selected_cam),
                                                   selected_cam.shape,
                                                   name='cam_mask')
            loss_lambda = tf.placeholder_with_default(0.0, (),
                                                      name='loss_lambda')

            masked_cam = tf.multiply(selected_cam, cam_mask, name='masked_cam')

            cam_loss = tf.multiply(tf.reduce_sum(masked_cam), loss_lambda,
                                   name='loss_cam_v')

        gamma = 2
        lbda = 0.25

        gamma_fl = tf.placeholder_with_default(2, (),name='gamma_fl')
        lbda_fl = tf.placeholder_with_default(0.25, (), name='lbda_fl')

        pond = tf.reduce_sum( lbda_fl * (1-self.pred) ** (gamma_fl) * self.targets, axis=1)
        ce_row = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.targets, logits=predictions)
        self.loss = tf.reduce_sum(pond * ce_row)


        # self.loss = cam_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(self.loss)

        # get accuracy
        prediction = tf.argmax(predictions, 1)
        equality = tf.equal(prediction, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))