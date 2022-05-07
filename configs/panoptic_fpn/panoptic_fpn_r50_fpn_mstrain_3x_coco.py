_base_ = './panoptic_fpn_r50_fpn_1x_coco.py'

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='RandomResize', img_scale=[(1333, 640), (1333, 800)]),
    dict(type='RandomFlip', prob=0.5),
    dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# TODO: Use RepeatDataset to speed up training
# training schedule for 3x
train_cfg = dict(by_epoch=True, max_epochs=36)
val_cfg = dict(interval=3)
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]