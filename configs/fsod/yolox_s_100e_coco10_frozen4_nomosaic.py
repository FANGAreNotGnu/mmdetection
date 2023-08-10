_base_ = './yolox_s_100e_coco10_frozen4.py'

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=99,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
