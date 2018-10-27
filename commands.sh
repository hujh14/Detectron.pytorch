CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml --load_detectron data/detectron_model/mask_rcnn_R-50-FPN_1x.pkl

CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset coco --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml --load_detectron data/detectron_model/mask_rcnn_R-50-FPN_1x.pkl --image_dir ../datasets/ade20k/images/validation

