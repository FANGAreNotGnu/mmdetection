_base_ = './yolox_s_200e_coco_base60.py'


# dataset settings
data_root = '/media/data/coco17/coco/'

# Example to use different file
val_dataloader = dict(dataset=dict(ann_file='annotations/instances_val2017_novel.json'))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017_novel.json')
test_evaluator = val_evaluator
