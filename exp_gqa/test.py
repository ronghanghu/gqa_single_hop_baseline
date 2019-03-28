import os
import numpy as np
import tensorflow as tf

from models_gqa.model import Model
from models_gqa.config import build_cfg_from_argparse
from util.gqa_train.data_reader import DataReader
import json

# Load config
cfg = build_cfg_from_argparse()

# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

# Data files
imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_VQA
scene_graph_file = cfg.SCENE_GRAPH_FILE % \
    cfg.TEST.SPLIT_VQA.replace('_balanced', '').replace('_all', '')
data_reader = DataReader(
    imdb_file, shuffle=False, one_pass=True, batch_size=cfg.TEST.BATCH_SIZE,
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
    num_vocab=num_vocab, num_choices=num_choices, is_training=False)

# Load snapshot
if cfg.TEST.USE_EMA:
    ema = tf.train.ExponentialMovingAverage(decay=0.9)  # decay doesn't matter
    var_names = {
        (ema.average_name(v) if v in model.params else v.op.name): v
        for v in tf.global_variables()}
else:
    var_names = {v.op.name: v for v in tf.global_variables()}
snapshot_file = cfg.TEST.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TEST.ITER)
print('loading model snapshot from %s' % snapshot_file)
snapshot_saver = tf.train.Saver(var_names)
snapshot_saver.restore(sess, snapshot_file)
print('Done')

# Write results
result_dir = cfg.TEST.RESULT_DIR % (cfg.EXP_NAME, cfg.TEST.ITER)
os.makedirs(result_dir, exist_ok=True)

# Run test
answer_correct, num_questions = 0, 0
if cfg.TEST.OUTPUT_VQA_EVAL_PRED:
    output_predictions = []
    answer_word_list = data_reader.batch_loader.answer_dict.word_list
    pred_file = os.path.join(
        result_dir, 'gqa_eval_preds_%s_%s_%08d.json' % (
            cfg.TEST.SPLIT_VQA, cfg.EXP_NAME, cfg.TEST.ITER))
for n_batch, batch in enumerate(data_reader.batches()):
    if 'answer_label_batch' not in batch:
        batch['answer_label_batch'] = -np.ones(
            len(batch['qid_list']), np.int32)
        if num_questions == 0:
            print('imdb has no answer labels. Using dummy labels.\n\n'
                  '**The final accuracy will be zero (no labels provided)**\n')

    vqa_scores_value = sess.run(model.vqa_scores, feed_dict={
        input_seq_batch: batch['input_seq_batch'],
        seq_length_batch: batch['seq_length_batch'],
        image_feat_batch: batch['image_feat_batch'],
        image_valid_batch: batch['image_valid_batch']})

    # compute accuracy
    vqa_labels = batch['answer_label_batch']
    vqa_predictions = np.argmax(vqa_scores_value, axis=1)
    answer_correct += np.sum(vqa_predictions == vqa_labels)
    num_questions += len(vqa_labels)
    accuracy = answer_correct / num_questions
    if n_batch % 20 == 0:
        print('exp: %s, iter = %d, accumulated accuracy on %s = %f (%d / %d)' %
              (cfg.EXP_NAME, cfg.TEST.ITER, cfg.TEST.SPLIT_VQA,
               accuracy, answer_correct, num_questions))

    if cfg.TEST.OUTPUT_VQA_EVAL_PRED:
        output_predictions.extend([
            {"questionId": qId, "prediction": answer_word_list[p]}
            for qId, p in zip(batch['qid_list'], vqa_predictions)])

with open(os.path.join(
        result_dir, 'vqa_results_%s.txt' % cfg.TEST.SPLIT_VQA), 'w') as f:
    print('\nexp: %s, iter = %d, final accuracy on %s = %f (%d / %d)' %
          (cfg.EXP_NAME, cfg.TEST.ITER, cfg.TEST.SPLIT_VQA,
           accuracy, answer_correct, num_questions))
    print('exp: %s, iter = %d, final accuracy on %s = %f (%d / %d)' %
          (cfg.EXP_NAME, cfg.TEST.ITER, cfg.TEST.SPLIT_VQA,
           accuracy, answer_correct, num_questions), file=f)

if cfg.TEST.OUTPUT_VQA_EVAL_PRED:
    with open(pred_file, 'w') as f:
        json.dump(output_predictions, f, indent=2)
    print('prediction file written to %s' % pred_file)
