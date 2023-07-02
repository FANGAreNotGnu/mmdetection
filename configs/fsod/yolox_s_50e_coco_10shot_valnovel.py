_base_ = './yolox_s_50e_coco_10shot.py'

# dataset settings
data_root = '/media/data/coco17/coco/'

# Example to use different file
val_dataloader = dict(dataset=dict(ann_file='annotations/instances_val2017_novel.json'))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017_novel.json')
test_evaluator = val_evaluator

# bash tools/dist_test.sh configs/fsod/yolox_s_50e_coco_10shot_valnovel.py /home/ubuntu/mmdetection/work_dirs/yolox_s_50e_coco_10shot/epoch_50.pth  2