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
        loss=dict(type='SmoothL1Loss', loss_weight=1.0),
        cal_acc=True,
    ))
