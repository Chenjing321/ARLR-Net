_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'fista'
work_dir = 'work_dir/pth/cso_compare/fista'

Phi_data_Name = 'data/sampling_matrix/a_phi_0_3.mat'
Qinit_Name = 'data/initial_matrix/Q_3.mat'

model = dict(
    type='Fista',
    LayerNo=7,
    Phi_data_Name=Phi_data_Name,
    Qinit_Name=Qinit_Name,
)
