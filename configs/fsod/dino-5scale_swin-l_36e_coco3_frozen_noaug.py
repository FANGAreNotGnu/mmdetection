_base_ = "./dino-5scale_swin-l_36e_coco_base60.py"

model = dict(
    backbone=dict(
        frozen_stages=4,),  # TODO
    bbox_head=dict(
        num_classes=20,  # TODO: change this in configs for novel train
        ),
    )

data_root = '/media/data/coco_fsod/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(640, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(500, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(640, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]


train_dataloader = dict(
    dataset=dict(
        data_root=data_root, ann_file='seed1/3shot_novel.json', 
        data_prefix=dict(img='train2017/'), pipeline=train_pipeline),)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val2017_novel.json',  # TODO: may need to change this in other configs
        data_prefix=dict(img='val2017/'),  # TODO: may need to change this in other configs
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017_novel.json',)
test_evaluator = val_evaluator

lr = 0.0001

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 150
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=max_epochs)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[int(max_epochs * 0.834)],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=max_epochs),)

load_from = '/home/ubuntu/mmdetection/work_dirs/dino-5scale_swin-l_36e_coco_base60/epoch_36.pth'
