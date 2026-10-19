dataset_type = 'NoiseDataset'

train_data_root = 'data/cso_data/train'
val_data_root = 'data/cso_data/val'
test_data_root = 'data/cso_data/test'

Phi_data_Name = 'data/sampling_matrix/phi_5.mat'
Qinit_Name = 'data/initial_matrix/Q_5.mat'

experiment_name = 'arlr_c5'
work_dir = 'work_dir/pth/cso_compare_c5/arlr'

model = dict(
    type='ARLR',
    LayerNo=6,
    Phi_data_Name=Phi_data_Name,
    Qinit_Name=Qinit_Name,
    block='ARLR_BasicBlock',
    lambda_weight=0.7,
    threshold_module='AdaptiveLSKThreshold',
    dynamic_conv_module='CorrectedDynamicKernelMixture',
    c=5,
    subpixel_offset=True,
    subpixel_offset_loss_weight=0.05,
    subpixel_offset_scale=0.5,
    subpixel_offset_beta=0.05,
    subpixel_offset_detach=True,
)

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=train_data_root,
        length=80000,
        c=5,
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
        c=5,
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
        c=5,
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
        c=5,
        subpixel_refine=True,
        subpixel_method='offset',
        subpixel_offset_limit=0.5,
    ),
]
test_evaluator = [
    dict(
        type='CSO_Metrics',
        brightness_threshold=50,
        c=5,
        subpixel_refine=True,
        subpixel_method='offset',
        subpixel_offset_limit=0.5,
    ),
    dict(
        type='Similarity',
        c=5,
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
