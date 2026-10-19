_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'fenet'
work_dir = 'work_dir/pth/cso_compare/fenet'

model = dict(
    type='FENet',
    scale=3,
    lst_channels=1,
    group=4,
    n_blocks=12,
)
