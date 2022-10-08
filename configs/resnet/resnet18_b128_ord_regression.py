_base_ = [
    '../_base_/datasets/quality_score_bs128_ensemble.py'
]

# model settings
model = dict(
    type='ImageRegression',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearRegHead',
        use_sigmoid=True,
        num_classes=1,
        in_channels=512,
        loss=dict(type='OrdinalRegressionLoss', K=5, loss_weight=1.0),
        cal_acc=True,
    ))
runner = dict(type='EpochBasedRunner', max_epochs=140)

# ==================================================
#           ../_base_/schedules/**
# ==================================================
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[40, 80, 120])

# ==================================================
#       to replace _base_/default_runtime.py
# ==================================================
# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
