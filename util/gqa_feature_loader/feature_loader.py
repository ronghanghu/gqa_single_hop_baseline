import h5py
import json
import numpy as np
import os.path as osp
from glob import glob


class SpatialFeatureLoader:
    def __init__(self, feature_dir):
        info_file = osp.join(feature_dir, 'gqa_spatial_info.json')
        with open(info_file) as f:
            self.all_info = json.load(f)

        num_files = len(glob(osp.join(feature_dir, 'gqa_spatial_*.h5')))
        h5_paths = [osp.join(feature_dir, 'gqa_spatial_%d.h5' % n)
                    for n in range(num_files)]
        self.h5_files = [h5py.File(path, 'r') for path in h5_paths]

    def __del__(self):
        for f in self.h5_files:
            f.close()

    def load_feature(self, imageId):
        info = self.all_info[imageId]
        file, idx = info['file'], info['idx']
        return self.h5_files[file]['features'][idx]


class ObjectsFeatureLoader:
    def __init__(self, feature_dir):
        info_file = osp.join(feature_dir, 'gqa_objects_info.json')
        with open(info_file) as f:
            self.all_info = json.load(f)

        num_files = len(glob(osp.join(feature_dir, 'gqa_objects_*.h5')))
        h5_paths = [osp.join(feature_dir, 'gqa_objects_%d.h5' % n)
                    for n in range(num_files)]
        self.h5_files = [h5py.File(path, 'r') for path in h5_paths]

    def __del__(self):
        for f in self.h5_files:
            f.close()

    def load_feature(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        feature = self.h5_files[file]['features'][idx]
        valid = get_valid(len(feature), num)
        return feature, valid

    def load_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        bbox = self.h5_files[file]['bboxes'][idx]
        valid = get_valid(len(bbox), num)
        return bbox, valid

    def load_feature_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        h5_file = self.h5_files[file]
        feature, bbox = h5_file['features'][idx], h5_file['bboxes'][idx]
        valid = get_valid(len(bbox), num)
        return feature, bbox, valid

    def load_normalized_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        bbox = self.h5_files[file]['bboxes'][idx]
        w, h = info['width'], info['height']
        normalized_bbox = bbox / [w, h, w, h]
        valid = get_valid(len(bbox), num)
        return normalized_bbox, valid

    def load_feature_normalized_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        h5_file = self.h5_files[file]
        feature, bbox = h5_file['features'][idx], h5_file['bboxes'][idx]
        w, h = info['width'], info['height']
        normalized_bbox = bbox / [w, h, w, h]
        valid = get_valid(len(bbox), num)
        return feature, normalized_bbox, valid


def get_valid(total_num, valid_num):
    valid = np.zeros(total_num, np.bool)
    valid[:valid_num] = True
    return valid
