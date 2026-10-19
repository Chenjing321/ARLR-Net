_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'gglnet'
work_dir = 'work_dir/pth/cso_compare/gglnet'

model = dict(
    type='GGLNet',
    in_channels=1,
    out_channels=1,
    adapter_size=64,
    output_size=33,
    output_scale=255.0,
)
