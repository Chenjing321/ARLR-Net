_base_ = '../_base_/datasets/img_dataset.py'

experiment_name = 'srfbn_c7'
work_dir = 'work_dir/pth/cso_compare_c7/srfbn'

model = dict(
    type='SRFBN',
    upscale_factor=7,
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
