dataset_type = 'NoiseDataset'

train_data_root = 'data/cso_data/train'
val_data_root = 'data/cso_data/val'
test_data_root = 'data/cso_data/test'

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=train_data_root,
        length=80000,
    ),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=val_data_root,
        length=10000,
    ),
    drop_last=True,
)

test_dataloader = dict(
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=test_data_root,
        length=10000,
    ),
    drop_last=True,
)

train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0001),
)

val_evaluator = [
    dict(
        type='CSO_Metrics',
        brightness_threshold=50,
        c=3,
        subpixel_refine=False,
    ),
]

test_evaluator = [
    dict(
        type='CSO_Metrics',
        brightness_threshold=50,
        c=3,
        subpixel_refine=False,
    ),
    dict(
        type='Similarity',
        c=3,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='cso_metric/mAP',
        rule='greater',
        max_keep_ckpts=1,
    ),
    visualization=dict(type='mmdet.DetVisualizationHook', draw=False),
)
