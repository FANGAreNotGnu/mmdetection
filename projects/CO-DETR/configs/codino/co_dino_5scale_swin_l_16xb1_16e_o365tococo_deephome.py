_base_ = ['co_dino_5scale_swin_l_16xb1_16e_o365tococo.py']

# Modify dataset related settings
data_root = '/media/deephome/data/'
metainfo = {
    'classes': [str(i) for i in range(151)],
    'palette': [(i, i, i) for i in range(151)],
}

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='deephome_train_labels.json',
        data_prefix=dict(img='train_images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='deephome_train_labels.json',
        data_prefix=dict(img='train_images/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'deephome_train_labels.json')
test_evaluator = val_evaluator

train_cfg = dict(val_interval=999)  # No Validation
