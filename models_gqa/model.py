import tensorflow as tf

from . import input_unit, single_hop


class Model:
    def __init__(self, input_seq_batch, seq_length_batch, image_feat_batch,
                 image_valid_batch, num_vocab, num_choices, is_training,
                 scope='model', reuse=None):
        """
        A naive VQA model with a single-hop attention
        """

        with tf.variable_scope(scope, reuse=reuse):
            q_encoding = input_unit.build_input_unit(
                input_seq_batch, seq_length_batch, num_vocab)
            self.single_hop = single_hop.SingleHop(
                image_feat_batch, q_encoding, image_valid_batch, num_choices)
            self.vqa_scores = self.single_hop.logits

            self.params = [
                v for v in tf.trainable_variables() if scope in v.op.name]
            self.l2_reg = tf.add_n(
                [tf.nn.l2_loss(v) for v in self.params
                 if v.op.name.endswith('weights')])
