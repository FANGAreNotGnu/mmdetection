_base_ = ["./yolox_s_200e_coco_base60.py"]

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))

train_dataloader = dict(
    batch_size=6,
    num_workers=2,)
val_dataloader = dict(
    batch_size=6,
    num_workers=2,)
test_dataloader = val_dataloader

# bash tools/dist_train.sh configs/fsod/yolox_l_200e_coco_base60.py 3 --auto-scale-lr
