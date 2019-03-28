import numpy as np
import tensorflow as tf

from .config import cfg


def build_input_unit(input_seq_batch, seq_length_batch, num_vocab,
                     scope='input_unit', reuse=None):
    """
    Encode the input question with a simple Bi-LSTM
    """
    with tf.variable_scope(scope, reuse=reuse):
        # word embedding
        if cfg.INIT_WRD_EMB_FROM_FILE:
            embed_mat_nparray = np.load(cfg.WRD_EMB_INIT_FILE)
            assert embed_mat_nparray.shape == (num_vocab, cfg.WRD_EMB_DIM)
            embed_mat = tf.get_variable(
                'embed_mat', initializer=embed_mat_nparray.astype(np.float32),
                trainable=(not cfg.WRD_EMB_FIXED))
        else:
            embed_mat = tf.get_variable(
                'embed_mat', [num_vocab, cfg.WRD_EMB_DIM],
                initializer=tf.initializers.random_normal(
                    stddev=np.sqrt(1. / cfg.WRD_EMB_DIM)),
                trainable=(not cfg.WRD_EMB_FIXED))
        embed_seq = tf.nn.embedding_lookup(embed_mat, input_seq_batch)

        # bidirectional LSTM
        assert cfg.ENC_DIM % 2 == 0, 'cfg.ENC_DIM must be a multiply of 2'
        lstm_dim = cfg.ENC_DIM
        cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_dim//2, name='basic_lstm_cell')
        cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_dim//2, name='basic_lstm_cell')
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, embed_seq, dtype=tf.float32,
            sequence_length=seq_length_batch, time_major=True)
        # concatenate the final hidden state of the forward and backward LSTM
        # for question representation
        q_encoding = tf.concat([states[0].h, states[1].h], axis=1)

    return q_encoding
