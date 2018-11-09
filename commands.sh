

# Train on 4 gpus
python tools/train_net_step.py --dataset ade20k --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml --use_tfboard

# Infer on im_list
CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py \
--dataset coco \
--cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
--load_detectron data/detectron_model/mask_rcnn_R-50-FPN_1x.pkl \
--im_dir data/ade20k/images \
--im_list data/ade20k/images/validation.txt \
--output_dir data/ade20k/predictions/maskrcnnc/

# Test coco
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
--load_detectron data/detectron_model/mask_rcnn_R-50-FPN_1x.pkl
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
--load_ckpt Outputs/e2e_mask_rcnn_R-50-FPN_1x/Oct27-19-46-13_visiongpu05_step/ckpt/model_step64984.pth
# Test ade20k
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset ade20k --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
--load_ckpt Outputs/e2e_mask_rcnn_R-50-FPN_1x/Nov05-23-51-06_visiongpu03_step/ckpt/model_step89999.pth