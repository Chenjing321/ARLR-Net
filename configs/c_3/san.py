_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'san'
work_dir = 'work_dir/pth/cso_compare/san'

model = dict(
    type='SAN',
    scale=3,
    n_colors=1,
    n_resgroups=10,
    n_resblocks=5,
    n_feats=64,
    kernel_size=3,
    reduction=16,
)
