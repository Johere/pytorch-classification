#!/usr/bin/env bash
set -x

ROOT=../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}

export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR=/dev/shm

EXP_NAME=${1-debug}  # qnet_exp2.2
TEST_SET=${2-ua_detrac_ens}  # ua_detrac_ens, boxcars_ens, ua_detrac_blur, boxcars_blur, ua_detrac_mag, boxcars_mag
GPUS=${3-3}

cfg=$(ls output/${EXP_NAME}/*.py)
CONFIG_FILE=${cfg}

ckpt=$(ls output/${EXP_NAME}/latest.pth)
CHECKPOINT=${ckpt}

JOB_NAME=test_${TEST_SET}_${EXP_NAME}
OUTPUT_DIR=output/${JOB_NAME}

mkdir -p ./logs

if [ ${TEST_SET} == "ua_detrac_ens" ]; then
    DATA_PREFIX=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/images
    ANN_FILE=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/ensemble_quality_score_weights_1.0_1.0_test.txt

elif [ ${TEST_SET} == "ua_detrac_blur" ]; then
    DATA_PREFIX=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/images
    ANN_FILE=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/trainval_laplace_var_normalize_test.txt

elif [ ${TEST_SET} == "ua_detrac_mag" ]; then
    DATA_PREFIX=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/images
    ANN_FILE=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_test.list

elif [ ${TEST_SET} == "boxcars_ens" ]; then
    DATA_PREFIX=/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/images
    ANN_FILE=/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/quality_score/dataset-splits/ensemble_quality_score_weights_1.0_1.0_test.txt

elif [ ${TEST_SET} == "boxcars_blur" ]; then
    DATA_PREFIX=/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/images
    ANN_FILE=/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/quality_score/dataset-splits/trainval_laplace_var_normalize_test.txt

elif [ ${TEST_SET} == "boxcars_mag" ]; then
    DATA_PREFIX=/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/images
    ANN_FILE=/mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_test.list
fi

echo 'start testing:' ${EXP_NAME} 'config:' ${CONFIG_FILE} 'checkpoint: ' ${CHECKPOINT}
python -u ${ROOT}/tools/test.py ${CONFIG_FILE} ${CHECKPOINT} \
    --metrics mae \
    --out ${OUTPUT_DIR} \
    --gpu-id ${GPUS} \
    --cfg-options \
    data.test.data_prefix=${DATA_PREFIX} \
    data.test.ann_file=${ANN_FILE} \
    2>&1 | tee logs/${JOB_NAME}.log

echo ${JOB_NAME} 'done.'

#    --eval mAP \
# --format-only \
