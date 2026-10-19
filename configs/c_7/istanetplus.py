_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'istanetplus_c7'
work_dir = 'work_dir/pth/cso_compare_c7/istanetplus'

Phi_data_Name = 'data/sampling_matrix/phi_7.mat'
Qinit_Name = 'data/initial_matrix/Q_7.mat'

model = dict(
    type='ISTANetplus',
    LayerNo=4,
    Phi_data_Name=Phi_data_Name,
    Qinit_Name=Qinit_Name,
    c=7,
)

train_dataloader = dict(dataset=dict(c=7))
val_dataloader = dict(dataset=dict(c=7))
test_dataloader = dict(dataset=dict(c=7))

val_evaluator = [
    dict(type='CSO_Metrics', brightness_threshold=50, c=7, subpixel_refine=False),
]

test_evaluator = [
    dict(type='CSO_Metrics', brightness_threshold=50, c=7, subpixel_refine=False),
    dict(type='Similarity', c=7),
]

train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=1)
