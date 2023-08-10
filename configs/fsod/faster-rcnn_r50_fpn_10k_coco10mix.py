_base_ = "./faster-rcnn_r50_fpn_10k_coco10_frozen4.py"

train_data_root = "/media/data/dad/cnet/experiments/coco10s1_512p/mix_n200_o0_m0_s1_p512_imprior"

train_dataloader = dict(
    dataset=dict(
        data_root=train_data_root, ann_file='annotation.json', 
        data_prefix=dict(img='images/')))

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=20000)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20000,
        by_epoch=False,
        milestones=[16000, 18000],
        gamma=0.1)
]

load_from = '/home/ubuntu/mmdetection/work_dirs/faster-rcnn_r50_fpn_2x_coco_base60/epoch_24.pth'

# CUDA_VISIBLE_DEVICES=3 python3 tools/train.py configs/fsod/faster-rcnn_r50_fpn_10k_coco10mix.py --auto-scale-lr --cfg-options randomness.seed=1
