_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'ctnet'
work_dir = 'work_dir/pth/cso_compare/ctnet'

model = dict(
    type='CTNET',
    scale=3,
    in_nc=1,
    out_nc=1,
    nf=64,
    unf=24,
    nb=16,
)
