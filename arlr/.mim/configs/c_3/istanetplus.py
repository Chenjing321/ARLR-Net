_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'istanetplus'
work_dir = 'work_dir/pth/cso_compare/istanetplus'

Phi_data_Name = 'data/sampling_matrix/a_phi_0_3.mat'
Qinit_Name = 'data/initial_matrix/Q_3.mat'

model = dict(
    type='ISTANetplus',
    LayerNo=9,
    Phi_data_Name=Phi_data_Name,
    Qinit_Name=Qinit_Name,
    c=3,
)
