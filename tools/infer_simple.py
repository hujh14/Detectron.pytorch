from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument('-l', '--im_list', type=str, default=None)
    parser.add_argument('-d', '--im_dir', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str,  default=None)

    args = parser.parse_args()

    return args


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset.startswith("ade20k"):
        dataset = datasets.get_ade_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()

    im_list = []
    with open(args.im_list) as f:
        im_list = f.read().splitlines()

    vis_dir = os.path.join(args.output_dir, "vis")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    detections = {}
    for i, im_name in enumerate(im_list):
        print("Mask-RCNN {} / {} : {}".format(i+1,len(im_list), im_name))
        img_path = os.path.join(args.im_dir, im_name)
        im = cv2.imread(img_path)
        assert im is not None

        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)
        detections[im_name] = (cls_boxes, cls_segms, cls_keyps)

        name, _ = os.path.splitext(os.path.basename(im_name))
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            name,
            vis_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )
    with open(os.path.join(args.output_dir, 'detections.pkl'), 'wb') as f:
        pickle.dump(detections, f)


if __name__ == '__main__':
    main()
