import threading
import queue
import numpy as np
import json

from util import text_processing
from util.positional_encoding import get_positional_encoding
from util.gqa_feature_loader.feature_loader import (
    SpatialFeatureLoader, ObjectsFeatureLoader)
from util.gqa_feature_loader.scene_graph_loader import SceneGraphFeatureLoader


class BatchLoaderGqa:
    def __init__(self, imdb, data_params):
        self.imdb = imdb
        self.data_params = data_params

        self.vocab_dict = text_processing.VocabDict(
            data_params['vocab_question_file'])
        self.T_encoder = data_params['T_encoder']

        # peek one example to see whether answer is in the data
        self.load_answer = ('answer' in self.imdb[0])
        # the answer dict is always loaded, regardless of self.load_answer
        self.answer_dict = text_processing.VocabDict(
            data_params['vocab_answer_file'])
        if not self.load_answer:
            print('imdb does not contain answers')

        self.load_spatial_feature = False
        self.load_objects_feature = False
        self.load_scene_graph_feature = False
        feature_type = data_params['feature_type']
        if feature_type == 'spatial':
            self.load_spatial_feature = True
        elif feature_type == 'objects':
            self.load_objects_feature = True
        elif feature_type == 'scene_graph':
            self.load_scene_graph_feature = True
        else:
            raise ValueError('Unknown feature type: %s' % feature_type)

        if self.load_spatial_feature:
            spatial_feature_dir = data_params['spatial_feature_dir']
            self.spatial_loader = SpatialFeatureLoader(spatial_feature_dir)
            # load one feature map to peek its size
            x = self.spatial_loader.load_feature(self.imdb[0]['imageId'])
            self.spatial_D, self.spatial_H, self.spatial_W = x.shape
            # positional encoding
            self.spatial_pos_enc_dim = data_params['spatial_pos_enc_dim']
            self.pos_enc = get_positional_encoding(
                self.spatial_H, self.spatial_W, self.spatial_pos_enc_dim)

        if self.load_objects_feature:
            objects_feature_dir = data_params['objects_feature_dir']
            self.objects_M = data_params['objects_max_num']
            self.objects_loader = ObjectsFeatureLoader(objects_feature_dir)
            # load one feature map to peek its size
            x, _ = self.objects_loader.load_feature(self.imdb[0]['imageId'])
            _, self.objects_D = x.shape
            self.bbox_tile_num = data_params['bbox_tile_num']

        if self.load_scene_graph_feature:
            scene_graph_file = data_params['scene_graph_file']
            vocab_name_file = data_params['vocab_name_file']
            vocab_attr_file = data_params['vocab_attr_file']
            self.objects_M = data_params['objects_max_num']
            self.scene_graph_loader = SceneGraphFeatureLoader(
                scene_graph_file, vocab_name_file, vocab_attr_file,
                max_num=self.objects_M)
            # load one feature map to peek its size
            x, _, _ = self.scene_graph_loader.load_feature_normalized_bbox(
                self.imdb[0]['imageId'])
            _, self.objects_D = x.shape
            self.bbox_tile_num = data_params['bbox_tile_num']

    def load_one_batch(self, sample_ids):
        actual_batch_size = len(sample_ids)
        input_seq_batch = np.zeros(
            (self.T_encoder, actual_batch_size), np.int32)
        seq_length_batch = np.zeros(actual_batch_size, np.int32)
        if self.load_spatial_feature:
            spatial_feat_batch = np.zeros(
                (actual_batch_size, self.spatial_D, self.spatial_H,
                 self.spatial_W), np.float32)
        if self.load_objects_feature or self.load_scene_graph_feature:
            objects_feat_batch = np.zeros(
                (actual_batch_size, self.objects_M, self.objects_D),
                np.float32)
            objects_bbox_batch = np.zeros(
                (actual_batch_size, self.objects_M, 4), np.float32)
            objects_valid_batch = np.zeros(
                (actual_batch_size, self.objects_M), np.bool)

        qid_list = [None]*actual_batch_size
        qstr_list = [None]*actual_batch_size
        imageid_list = [None]*actual_batch_size
        if self.load_answer:
            answer_label_batch = np.zeros(actual_batch_size, np.int32)
        for n in range(len(sample_ids)):
            iminfo = self.imdb[sample_ids[n]]
            question_str = iminfo['question']
            question_tokens = text_processing.tokenize(question_str)
            if len(question_tokens) > self.T_encoder:
                print('data reader: truncating question:\n\t' + question_str)
                question_tokens = question_tokens[:self.T_encoder]
            question_inds = [
                self.vocab_dict.word2idx(w) for w in question_tokens]
            seq_length = len(question_inds)
            input_seq_batch[:seq_length, n] = question_inds
            seq_length_batch[n] = seq_length

            if self.load_spatial_feature:
                feature = self.spatial_loader.load_feature(iminfo['imageId'])
                spatial_feat_batch[n:n+1] = feature
            if self.load_objects_feature:
                feature, normalized_bbox, valid = \
                    self.objects_loader.load_feature_normalized_bbox(
                        iminfo['imageId'])
                objects_feat_batch[n:n+1] = feature
                objects_bbox_batch[n:n+1] = normalized_bbox
                objects_valid_batch[n:n+1] = valid
            if self.load_scene_graph_feature:
                feature, normalized_bbox, valid = \
                    self.scene_graph_loader.load_feature_normalized_bbox(
                        iminfo['imageId'])
                objects_feat_batch[n:n+1] = feature
                objects_bbox_batch[n:n+1] = normalized_bbox
                objects_valid_batch[n:n+1] = valid

            qid_list[n] = iminfo['questionId']
            qstr_list[n] = question_str
            imageid_list[n] = iminfo['imageId']
            if self.load_answer:
                answer_idx = self.answer_dict.word2idx(iminfo['answer'])
                answer_label_batch[n] = answer_idx
        batch = dict(input_seq_batch=input_seq_batch,
                     seq_length_batch=seq_length_batch,
                     qid_list=qid_list, qstr_list=qstr_list,
                     imageid_list=imageid_list)

        # 'image_feat_batch': N x H x W x C tf.float32 image features
        #   When using objects, then H = 1 & W = 100
        # 'image_valid_batch': N x H x W tf.float32, indicating whether
        #   each feature location is real (1) or padding (0)
        #   When using objects, image_valid_batch is 1 on objects & 0 otherwise
        if self.load_spatial_feature:
            # NCHW -> NHWC
            spatial_feat_batch = spatial_feat_batch.transpose((0, 2, 3, 1))
            batch['spatial_feat_batch'] = spatial_feat_batch
            # add positional embedding to the image features
            pos_enc_tile = np.tile(
                self.pos_enc, (len(spatial_feat_batch), 1, 1, 1))
            image_feat_batch = np.concatenate(
                 (spatial_feat_batch, pos_enc_tile), axis=-1)
            image_valid_batch = np.ones(image_feat_batch.shape[:3], np.float32)
            batch['image_feat_batch'] = image_feat_batch
            batch['image_valid_batch'] = image_valid_batch
        if self.load_objects_feature or self.load_scene_graph_feature:
            batch['objects_feat_batch'] = objects_feat_batch
            batch['objects_bbox_batch'] = objects_bbox_batch
            # add bounding boxes to the object features
            # tile bbox to roughly match the l2 norm of R-CNN features
            objects_bbox_tile = np.tile(
                objects_bbox_batch, (1, 1, self.bbox_tile_num))
            image_feat_batch = np.concatenate(
                (objects_feat_batch, objects_bbox_tile), axis=-1)
            image_feat_batch = image_feat_batch[:, np.newaxis, :, :]
            image_valid_batch = objects_valid_batch[:, np.newaxis, :] * 1.
            batch['image_feat_batch'] = image_feat_batch
            batch['image_valid_batch'] = image_valid_batch
        if self.load_answer:
            batch['answer_label_batch'] = answer_label_batch
        return batch


class DataReader:
    def __init__(self, json_file, shuffle=True, one_pass=False, prefetch_num=8,
                 **kwargs):
        print('Loading imdb from %s' % json_file)
        with open(json_file) as f:
            raw_data = json.load(f)
            qIds = sorted(raw_data)
            for qId, q in raw_data.items():
                q['questionId'] = qId
            imdb = [raw_data[qId] for qId in qIds]
        print('Done')
        self.imdb = imdb
        self.shuffle = shuffle
        self.one_pass = one_pass
        self.prefetch_num = prefetch_num
        self.data_params = kwargs

        # Vqa data loader
        self.batch_loader = BatchLoaderGqa(self.imdb, self.data_params)

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_num)
        self.prefetch_thread = threading.Thread(
            target=_run_prefetch, args=(
                self.prefetch_queue, self.batch_loader, self.imdb,
                self.shuffle, self.one_pass, self.data_params))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def batches(self):
        while True:
            # Get a batch from the prefetching queue
            batch = self.prefetch_queue.get(block=True)
            if batch is None:
                assert(self.one_pass)
                print('data reader: one pass finished')
                raise StopIteration()
            yield batch


def _run_prefetch(prefetch_queue, batch_loader, imdb, shuffle, one_pass,
                  data_params):
    num_samples = len(imdb)
    batch_size = data_params['batch_size']

    n_sample = 0
    fetch_order = np.arange(num_samples)
    while True:
        # Shuffle the sample order for every epoch
        if n_sample == 0 and shuffle:
            fetch_order = np.random.permutation(num_samples)

        # Load batch from file
        # note that len(sample_ids) <= batch_size, not necessarily equal
        sample_ids = fetch_order[n_sample:n_sample+batch_size]
        batch = batch_loader.load_one_batch(sample_ids)
        prefetch_queue.put(batch, block=True)

        n_sample += len(sample_ids)
        if n_sample >= num_samples:
            # Put in a None batch to indicate a whole pass is over
            if one_pass:
                prefetch_queue.put(None, block=True)
            n_sample = 0
