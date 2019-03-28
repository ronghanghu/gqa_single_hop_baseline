import tensorflow as tf
from tensorflow import convert_to_tensor as to_T, newaxis as ax

from util.cnn import (
    conv_layer as conv, fc_layer as fc, fc_relu_layer as fc_relu)
from .config import cfg


class SingleHop:
    def __init__(self, images, q_encoding, image_valid_batch, num_choices,
                 scope='single_hop', reuse=None):

        x_loc = self.loc_init(images, reuse=reuse)

        with tf.variable_scope(scope, reuse=reuse):
            x_loc_shape = tf.shape(x_loc)
            B, H, W = x_loc_shape[0], x_loc_shape[1], x_loc_shape[2]
            dim = x_loc.get_shape().as_list()[-1]  # static shape

            # attention over x_loc
            proj_q = fc('fc_q_map1', q_encoding, output_dim=dim)[:, ax, ax, :]
            interactions = tf.nn.l2_normalize(x_loc * proj_q, axis=-1)
            raw_att = conv('conv_att_score', interactions, kernel_size=1,
                           stride=1, output_dim=1)
            raw_att = tf.reshape(raw_att, to_T([B, H*W]))  # (N, H*W)
            valid_mask = tf.reshape(image_valid_batch, tf.shape(raw_att))
            raw_att = raw_att * valid_mask - 1e18 * (1-valid_mask)
            att = tf.nn.softmax(raw_att, axis=-1)  # (N, H*W)

            # collect attended image feature
            x_att = tf.matmul(
                tf.reshape(att, to_T([B, 1, H*W])),
                tf.reshape(x_loc, to_T([B, H*W, dim])))  # (N, 1, D_kb)
            x_att = tf.reshape(x_att, to_T([B, dim]))  # (N, D_kb)

            # VQA classification
            eQ = fc('fc_q_map2', q_encoding, output_dim=dim)
            if cfg.OUT_QUESTION_MUL:
                features = tf.concat([x_att, eQ, x_att*eQ], axis=-1)
            else:
                features = tf.concat([x_att, eQ], axis=-1)

            fc1 = fc_relu(
                'fc_hidden', features, output_dim=cfg.OUT_CLASSIFIER_DIM)
            logits = fc('fc_scores', fc1, output_dim=num_choices)
            self.logits = logits

    def loc_init(self, images, scope='kb_batch', reuse=None):
        """
        Linearly transform the input features to a fixed dimension MODEL.KB_DIM
        """
        with tf.variable_scope(scope, reuse=reuse):
            if cfg.STEM_NORMALIZE:
                images = tf.nn.l2_normalize(images, axis=-1)

            # apply a single layer convnet
            conv1 = conv('conv1', images, kernel_size=1, stride=1,
                         output_dim=cfg.LOC_DIM)
        return conv1
