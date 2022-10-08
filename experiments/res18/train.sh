#!/usr/bin/env bash
set -x

ROOT=../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}

export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR=/dev/shm
#export CUDA_VISIBLE_DEVICES='0,1'
#GPUS=${3-'0'}
#export CUDA_VISIBLE_DEVICES=${GPUS}

EXP_NAME=${1-debug}
GPUID=${2-1}
CONFIG_FILE=${3-${ROOT}/configs/resnet/resnet18_interfc_regression.py}


OUTPUT_DIR=output/${EXP_NAME}
# for safety
if [ -d ${OUTPUT_DIR} ]; then
    echo "job:" ${EXP_NAME} "exists before, 3 seconds for your withdrawing..."
    sleep 3
    rm -rf ${OUTPUT_DIR}
fi

mkdir -p ${OUTPUT_DIR}

echo 'start training:' ${EXP_NAME} 'config:' ${CONFIG_FILE}
python ${ROOT}/tools/train.py \
    ${CONFIG_FILE} \
    --gpu-ids ${GPUID} \
    --work-dir=${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/train.log

echo ${EXP_NAME} 'done.'

