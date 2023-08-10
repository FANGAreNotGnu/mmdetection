_base_ = "./faster-rcnn_r50_fpn_2x_coco_base60.py"

model = dict(
    backbone=dict(
        frozen_stages=4,  # TODO
    ),
    roi_head=dict(bbox_head=dict(num_classes=20)))

data_root = '/media/data/coco_fsod/'
dataset_type = 'CocoDataset'

img_scale=(1333, 800)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)), # The image will be enlarged by 4 times after Mosaic processing,so we use affine transformation to restore the image size.
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    #dict(
    #    type='Pad',
    #    pad_to_square=True,
    #    pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataset = dict(
    _delete_ = True, # remove unnecessary Settings
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='seed1/10shot_novel.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=None),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_val2017_novel.json',  # TODO
        pipeline=test_pipeline,
        )
    )
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017_novel.json',  # TODO
    classwise=True,
    )
test_evaluator = val_evaluator


train_cfg = dict(val_interval=24)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0005, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

load_from = '/home/ubuntu/mmdetection/work_dirs/faster-rcnn_r50_fpn_2x_coco_base60/epoch_24.pth'
