from core.config import cfg

import _init_paths  # pylint: disable=unused-import
from datasets.dataset_catalog import *

_DATA_DIR = cfg.DATA_DIR

DATASETS["ade20k_train"] = {
    IM_DIR:
        _DATA_DIR + '/ade20k/images',
    ANN_FN:
        _DATA_DIR + '/ade20k/annotations/instances_ade20k_train.json',
}
DATASETS["ade20k_val"] = {
    IM_DIR:
        _DATA_DIR + '/ade20k/images',
    ANN_FN:
        _DATA_DIR + '/ade20k/annotations/instances_ade20k_val.json',
}
