_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'srfbn'
work_dir = 'work_dir/pth/cso_compare/srfbn'

model = dict(
    type='SRFBN',
    upscale_factor=3,
    in_channels=1,
    out_channels=1,
    num_features=32,
    num_steps=1,
    num_groups=3,
    act_type='prelu',
    norm_type=None,
)
