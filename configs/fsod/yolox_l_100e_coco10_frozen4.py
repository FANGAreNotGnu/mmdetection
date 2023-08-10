_base_ = ["./yolox_l_200e_coco_base60.py"]

# model settings
model = dict(bbox_head=dict(num_classes=20))

data_root = '/media/data/coco_fsod/'
dataset_type = 'CocoDataset'

train_dataset = dict(
    dataset=dict(
        ann_file='seed1/10shot_novel.json',  # TODO
        )
    )

train_dataloader = dict(dataset=train_dataset,)

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_val2017_novel.json',  # TODO
        )
    )
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017_novel.json',  # TODO
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

load_from = '/home/ubuntu/mmdetection/work_dirs/yolox_l_200e_coco_base60/epoch_200.pth'

# CUDA_VISIBLE_DEVICES=4 python3 tools/train.py configs/fsod/yolox_l_100e_coco10_frozen4.py --auto-scale-lr --cfg-options randomness.seed=1 test_evaluator.classwise=True

