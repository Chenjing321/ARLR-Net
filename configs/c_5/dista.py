_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'dista_c5'
work_dir = 'work_dir/pth/cso_compare_c5/dista'

Phi_data_Name = 'data/sampling_matrix/phi_5.mat'
Qinit_Name = 'data/initial_matrix/Q_5.mat'

model = dict(
    type='DISTA',
    LayerNo=6,
    Phi_data_Name=Phi_data_Name,
    Qinit_Name=Qinit_Name,
    block='DIST_BasicBlock',
    lambda_weight=0.7,
    c=5,
)

train_dataloader = dict(dataset=dict(c=5))
val_dataloader = dict(dataset=dict(c=5))
test_dataloader = dict(dataset=dict(c=5))

val_evaluator = [
    dict(type='CSO_Metrics', brightness_threshold=50, c=5, subpixel_refine=False),
]

test_evaluator = [
    dict(type='CSO_Metrics', brightness_threshold=50, c=5, subpixel_refine=False),
    dict(type='Similarity', c=5),
]
