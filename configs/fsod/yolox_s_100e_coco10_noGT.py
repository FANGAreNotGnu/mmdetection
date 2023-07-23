_base_ = './yolox_s_100e_coco10_frozen4.py'

test_data_root = "/media/data/dad/cnet/experiments/coco10s1_512p/mix_n2000_o0_m0_s1_p512_pbe_pfa_csl_a10/"
test_ann_file = "annotation_noGT.json"
test_prefix = "images/"

train_data_root = '/media/data/coco_fsod/'
train_ann_file = "seed1/10shot_novel.json"
train_prefix = "train2017/"

dataset_type = 'CocoDataset'

train_dataset = dict(
    dataset=dict(
        data_root=train_data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_prefix),  # TODO
        )
    )

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    dataset=dict(
        data_root=test_data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=test_prefix),  # TODO
        )
    )
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=test_data_root + test_ann_file,  # TODO
    classwise=True,
    )
test_evaluator = val_evaluator

# training settings
max_epochs = 100
num_last_epochs = 10
interval = 100

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.005
#weight_decay = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
        #_delete_=True, type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)


load_from = '/home/ubuntu/mmdetection/work_dirs/yolox_s_200e_coco_base60/epoch_200.pth'

# CUDA_VISIBLE_DEVICES=6 python3 tools/train.py configs/fsod/yolox_s_100e_coco10_noGT.py --auto-scale-lr --cfg-options randomness.seed=1 test_evaluator.classwise=True
# bash tools/dist_train.sh configs/fsod/yolox_s_50e_coco_10shot.py 3 --auto-scale-lr
