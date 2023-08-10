_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=60)))

data_root = '/media/data/coco_fsod/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root, ann_file='annotations/instances_train2017_base.json', 
        data_prefix=dict(img='train2017/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val2017_base.json',  # TODO: may need to change this in other configs
        data_prefix=dict(img='val2017/'),  # TODO: may need to change this in other configs
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017_base.json',)
test_evaluator = val_evaluator
