
import _init_paths  # pylint: disable=unused-import
from datasets.dataset_catalog import *

DATASETS["ade20k_train"] = {
    IM_DIR:
        _DATA_DIR + '/ade20k/images/training',
    ANN_FN:
        _DATA_DIR + '/ade20k/annotations/annotations_instance/ade20k_train_instance.json',
}
DATASETS["ade20k_val"] = {
    IM_DIR:
        _DATA_DIR + '/ade20k/images/validation',
    ANN_FN:
        _DATA_DIR + '/ade20k/annotations/annotations_instance/ade20k_val_instance.json',
}