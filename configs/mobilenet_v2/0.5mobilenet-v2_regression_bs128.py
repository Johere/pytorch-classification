_base_ = [
    '../_base_/datasets/quality_score_bgr_bs128_mag.py'
]

# model settings
model = dict(
    type='ImageRegression',
    backbone=dict(type='MobileNetV2', widen_factor=0.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearRegHead',
        use_sigmoid=True,
        num_classes=1,
        in_channels=1280,
        loss=dict(type='SmoothL1Loss', loss_weight=1.0),
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

