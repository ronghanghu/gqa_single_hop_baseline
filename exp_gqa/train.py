import os
import numpy as np
import tensorflow as tf

from models_gqa.model import Model
from models_gqa.config import build_cfg_from_argparse
from util.gqa_train.data_reader import DataReader


# Load config
cfg = build_cfg_from_argparse()

# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

# Data files
imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
scene_graph_file = cfg.SCENE_GRAPH_FILE % \
    cfg.TRAIN.SPLIT_VQA.replace('_balanced', '').replace('_all', '')
data_reader = DataReader(
    imdb_file, shuffle=True, one_pass=False, batch_size=cfg.TRAIN.BATCH_SIZE,
    T_encoder=cfg.T_ENCODER,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
    feature_type=cfg.FEAT_TYPE,
    spatial_feature_dir=cfg.SPATIAL_FEATURE_DIR,
    objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
    objects_max_num=cfg.W_FEAT,
    scene_graph_file=scene_graph_file,
    vocab_name_file=cfg.VOCAB_NAME_FILE,
    vocab_attr_file=cfg.VOCAB_ATTR_FILE,
    spatial_pos_enc_dim=cfg.SPATIAL_POS_ENC_DIM,
    bbox_tile_num=cfg.BBOX_TILE_NUM)
num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
num_choices = data_reader.batch_loader.answer_dict.num_vocab

# Inputs and model
input_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(
    tf.float32, [None, cfg.H_FEAT, cfg.W_FEAT, cfg.D_FEAT])
image_valid_batch = tf.placeholder(
    tf.float32, [None, cfg.H_FEAT, cfg.W_FEAT])
model = Model(
    input_seq_batch, seq_length_batch, image_feat_batch, image_valid_batch,
    num_vocab=num_vocab, num_choices=num_choices, is_training=True)

# Loss function
answer_label_batch = tf.placeholder(tf.int32, [None])
loss_type = cfg.TRAIN.LOSS_TYPE
if loss_type == 'softmax':
    loss_vqa = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model.vqa_scores, labels=answer_label_batch))
elif loss_type == 'sigmoid':
    loss_vqa = tf.reduce_mean(
        tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=model.vqa_scores,
                labels=tf.one_hot(answer_label_batch, num_choices)),
            axis=-1))
else:
    raise Exception('Unknown loss type: %s' % loss_type)
loss_train = loss_vqa
loss_total = loss_train + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg

# Train with Adam
solver = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.SOLVER.LR)
solver_op = solver.minimize(loss_total)
# Save moving average of parameters
ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMA_DECAY)
ema_op = ema.apply(model.params)
with tf.control_dependencies([solver_op]):
    train_op = tf.group(ema_op)

# Save snapshot
snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
os.makedirs(snapshot_dir, exist_ok=True)
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
if cfg.TRAIN.START_ITER > 0:
    snapshot_file = os.path.join(snapshot_dir, "%08d" % cfg.TRAIN.START_ITER)
    print('resume training from %s' % snapshot_file)
    snapshot_saver.restore(sess, snapshot_file)
else:
    sess.run(tf.global_variables_initializer())
# Save config
np.save(os.path.join(snapshot_dir, 'cfg.npy'), np.array(cfg))

# Write summary to TensorBoard
log_dir = cfg.TRAIN.LOG_DIR % cfg.EXP_NAME
os.makedirs(log_dir, exist_ok=True)
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
loss_vqa_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])
summary_trn = []
summary_trn.append(tf.summary.scalar("loss/vqa", loss_vqa_ph))
summary_trn.append(tf.summary.scalar("eval/vqa/accuracy", accuracy_ph))
log_step_trn = tf.summary.merge(summary_trn)

# Run training
avg_accuracy, accuracy_decay = 0., 0.99
for n_batch, batch in enumerate(data_reader.batches()):
    n_iter = n_batch + cfg.TRAIN.START_ITER
    if n_iter >= cfg.TRAIN.MAX_ITER:
        break

    feed_dict = {input_seq_batch: batch['input_seq_batch'],
                 seq_length_batch: batch['seq_length_batch'],
                 image_feat_batch: batch['image_feat_batch'],
                 image_valid_batch: batch['image_valid_batch'],
                 answer_label_batch: batch['answer_label_batch']}
    vqa_scores_value, loss_vqa_value, _ = sess.run(
            (model.vqa_scores, loss_vqa, train_op), feed_dict)

    # compute accuracy
    vqa_labels = batch['answer_label_batch']
    vqa_predictions = np.argmax(vqa_scores_value, axis=1)
    accuracy = np.mean(vqa_predictions == vqa_labels)
    avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)

    # Print and add to TensorBoard summary
    if (n_iter+1) % cfg.TRAIN.LOG_INTERVAL == 0:
        print("exp: %s, iter = %d\n\t" % (cfg.EXP_NAME, n_iter+1) +
              "loss (vqa) = %f\n\t" % (loss_vqa_value) +
              "accuracy (current batch) = %f, "
              "accuracy (running average) = %f" % (accuracy, avg_accuracy))
        summary = sess.run(log_step_trn, {loss_vqa_ph: loss_vqa_value,
                                          accuracy_ph: avg_accuracy})
        log_writer.add_summary(summary, n_iter+1)

    # Save snapshot
    if ((n_iter+1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
            (n_iter+1) == cfg.TRAIN.MAX_ITER):
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter+1))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file)
