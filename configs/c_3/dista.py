_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'dista_c3'
work_dir = 'work_dir/pth/cso_compare/dista_c3'

Phi_data_Name = 'data/sampling_matrix/a_phi_0_3.mat'
Qinit_Name = 'data/initial_matrix/Q_3.mat'

model = dict(
    type='DISTA',
    LayerNo=6,
    Phi_data_Name=Phi_data_Name,
    Qinit_Name=Qinit_Name,
    block='DIST_BasicBlock',
    lambda_weight=0.7,
    c=3,
)

val_evaluator = [
    dict(type='CSO_Metrics', brightness_threshold=50, c=3, subpixel_refine=False),
]

test_evaluator = [
    dict(type='CSO_Metrics', brightness_threshold=50, c=3, subpixel_refine=False),
    dict(type='Similarity', c=3),
]
