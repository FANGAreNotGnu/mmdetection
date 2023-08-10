_base_ = "./dino-4scale_r50_36e_coco_base60.py"

model = dict(bbox_head=dict(num_classes=20))  # 100 for DeformDETR

data_root = '/media/data/voc_fsod/'
dataset_type = 'CocoDataset'
METAINFO = {
    'classes':
    ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
     'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    # palette is a list of color tuples, which is used for visualization.
    'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
                (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
                (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]
}

train_dataloader = dict(
    dataset=dict(
        data_root=data_root, ann_file='annotations/pascal_trainval0712.json', 
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False),
        metainfo=METAINFO,))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/pascal_test2007.json',  # TODO: may need to change this in other configs
        data_prefix=dict(img='images/'),  # TODO: may need to change this in other configs
        metainfo=METAINFO,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/pascal_test2007.json',)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
