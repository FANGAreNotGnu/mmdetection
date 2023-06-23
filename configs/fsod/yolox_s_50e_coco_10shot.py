_base_ = './yolox_s_300e_coco_base60.py'

data_root = '/media/data/coco17/coco/'
dataset_type = 'CocoDataset'

train_dataset = dict(
    dataset=dict(
        ann_file='seed1/10shot.json',  # TODO
        data_prefix=dict(img='train2017/'),  # TODO
        )
    )

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_val2017.json',  # TODO
        data_prefix=dict(img='val2017/'),  # TODO
        )
    )
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',  # TODO
    )
test_evaluator = val_evaluator

# training settings
max_epochs = 50
num_last_epochs = 5
interval = 5

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.001

load_from = '/home/ubuntu/mmdetection/work_dirs/yolox_s_30e_coco_base60/epoch_300.pth'


# bash tools/dist_train.sh configs/fsod/yolox_s_50e_coco_10shot.py 3 --auto-scale-lr
