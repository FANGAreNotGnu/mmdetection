_base_ = './yolox_s_200e_coco_base60.py'

num_classes = 31
data_root = '/media/data/odinw/others/plantdoc/'
classes = ('leaves','Apple Scab Leaf','Apple leaf','Apple rust leaf','Bell_pepper leaf','Bell_pepper leaf spot','Blueberry leaf','Cherry leaf','Corn Gray leaf spot','Corn leaf blight','Corn rust leaf','Peach leaf','Potato leaf','Potato leaf early blight','Potato leaf late blight','Raspberry leaf','Soyabean leaf','Soybean leaf','Squash Powdery mildew leaf','Strawberry leaf','Tomato Early blight leaf','Tomato Septoria leaf spot','Tomato leaf','Tomato leaf bacterial spot','Tomato leaf late blight','Tomato leaf mosaic virus','Tomato leaf yellow virus','Tomato mold leaf','Tomato two spotted spider mites leaf','grape leaf','grape leaf black rot')
palette = [(0,0,0),(1,3,7),(2,6,14),(3,9,21),(4,12,28),(5,15,35),(6,18,42),(7,21,49),(8,24,56),(9,27,63),(10,30,70),(11,33,77),(12,36,84),(13,39,91),(14,42,98),(15,45,105),(16,48,112),(17,51,119),(18,54,126),(19,57,133),(20,60,140),(21,63,147),(22,66,154),(23,69,161),(24,72,168),(25,75,175),(26,78,182),(27,81,189),(28,84,196),(29,87,203),(30,90,210)]
train_ann_file = 'train/_annotations.coco.json'
train_img_folder = 'train'
test_ann_file = 'test/_annotations.coco.json'
test_img_folder = 'test'
max_epochs = 300

# model settings
model = dict(bbox_head=dict(num_classes=num_classes,))

dataset_type = 'CocoDataset'
METAINFO = {
    'classes': classes,
    # palette is a list of color tuples, which is used for visualization.
    'palette': palette,
}

train_dataset = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,  # TODO
        data_prefix=dict(img=train_img_folder),  # TODO: may need to change this in other configs
        metainfo=METAINFO,
        )
    )

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=test_ann_file,  # TODO
        data_prefix=dict(img=test_img_folder),  # TODO: may need to change this in other configs
        metainfo=METAINFO,
        )
    )
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + test_ann_file,  # TODO
    )
test_evaluator = val_evaluator

# training settings
max_epochs = max_epochs
num_last_epochs = int(0.05 * max_epochs)
interval = max_epochs

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.01
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


#sload_from = '/home/ubuntu/mmdetection/work_dirs/yolox_s_200e_coco_base60/epoch_200.pth'

# CUDA_VISIBLE_DEVICES=6 python3 tools/train.py configs/fsod/yolox_s_100e_coco1_frozen4.py --auto-scale-lr --cfg-options randomness.seed=1
# bash tools/dist_train.sh configs/fsod/yolox_s_50e_coco_10shot.py 3 --auto-scale-lr
