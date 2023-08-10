_base_ = "./dino-5scale_swin-l_36e_coco30_frozen.py"

#train_data_root = '/media/data/dad/cnet/experiments/coco10novel/mix_n2000_o1_s1_p640_pfa_csl_p20_pfb_csl40'  # change this for different synthetic strategy
train_data_root = "/media/data/dad/cnet/experiments/coco30s1_512p/mix_n2500-150_dfsNone_o0_m0_s1_HED_p512_imprior"
dataset_type = 'CocoDataset'

train_dataloader = dict(
    dataset=dict(
        data_root=train_data_root,
        ann_file='annotation.json',  # TODO
        data_prefix=dict(img='images/'),  # TODO: may need to change this in other configs
        ))

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
max_epochs = 50
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
