CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml --load_detectron data/detectron_model/mask_rcnn_R-50-FPN_1x.pkl

python tools/train_net_step.py --dataset ade20k --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml --use_tfboard

CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py \
--dataset coco \
--cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
--load_detectron data/detectron_model/mask_rcnn_R-50-FPN_1x.pkl \
--im_dir data/places/images \
--im_list data/places/im_list/images0.txt \
--output_dir data/places/predictions/mask_rcnn_coco/images0