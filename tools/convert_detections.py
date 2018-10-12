from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import pickle

from pycocotools.coco import COCO

import _init_paths
import utils.vis as vis_utils

def process(detections):
    coco = COCO()
    for im in detections:
        file_name = os.path.basename(im)
        cls_boxes, cls_segms, cls_keyps = detections[im]
        boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
        print(filename, segms, classes)
        break

detection_fn = "data/ade20k/predictions/ade20k_val_maskrcnn_coco/detections.pkl"
with open(detection_fn, 'rb') as f:
    detections = pickle.load(f)
    process(detections)

