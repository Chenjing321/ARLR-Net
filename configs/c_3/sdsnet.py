_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'sdsnet'
work_dir = 'work_dir/pth/cso_compare/sdsnet'

model = dict(
    type='SDSNet',
    in_channels=1,
    out_channels=1,
    base_channel=32,
    patch_sizes=(16, 8, 4),
    num_layers=4,
    adapter_size=64,
    output_size=33,
    output_scale=255.0,
)
