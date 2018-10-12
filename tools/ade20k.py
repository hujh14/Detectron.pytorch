from core.config import cfg

import _init_paths  # pylint: disable=unused-import
from datasets.dataset_catalog import *

_DATA_DIR = cfg.DATA_DIR

DATASETS["ade20k_train"] = {
    IM_DIR:
        _DATA_DIR + '/ade20k/images/training',
    ANN_FN:
        _DATA_DIR + '/ade20k/annotations/annotations_instance/ade20k_train_annotations.json',
}
DATASETS["ade20k_val"] = {
    IM_DIR:
        _DATA_DIR + '/ade20k/images/validation',
    ANN_FN:
        _DATA_DIR + '/ade20k/annotations/annotations_instance/ade20k_val_annotations.json',
}
