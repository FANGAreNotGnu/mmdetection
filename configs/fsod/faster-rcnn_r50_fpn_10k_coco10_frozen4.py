_base_ = "./faster-rcnn_r50_fpn_2x_coco_base60.py"

model = dict(
    backbone=dict(
        frozen_stages=4,  # TODO
    ),
    roi_head=dict(bbox_head=dict(num_classes=20)))

data_root = '/media/data/coco_fsod/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root, ann_file='seed1/10shot_novel.json', 
        data_prefix=dict(img='train2017/')))
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

'''
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
'''
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=20000,  # 36 epochs
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
