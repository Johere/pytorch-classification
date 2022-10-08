# dataset settings
dataset_type='QAImage'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file='/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/mini_train.list',
        data_prefix=[
                        '/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/images',
                        '/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/images',
                    ],
        ann_file=[
                    '/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_train.list',
                    '/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_train.list',
                ],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file='/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/mini_train.list',
        data_prefix=[
                        '/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/images',
                        # '/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/images',
                    ],
        ann_file=[
                    '/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_test_sparse.list',
                    # '/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_test.list',
                ],
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        # ann_file='/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/mini_train.list',
        data_prefix=[
                        '/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/images',
                        # '/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/images',
                    ],
        ann_file=[
                    '/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_test_sparse.list',
                    # '/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_test.list',
                ],
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mae')
