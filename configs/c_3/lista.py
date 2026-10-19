_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'lista'
work_dir = 'work_dir/pth/cso_compare/lista'

Phi_data_Name = 'data/sampling_matrix/a_phi_0_3.mat'
Qinit_Name = 'data/initial_matrix/Q_3.mat'

model = dict(
    type='LISTA',
    LayerNo=16,
    Phi_data_Name=Phi_data_Name,
    Qinit_Name=Qinit_Name,
    lam=0.4,
)
