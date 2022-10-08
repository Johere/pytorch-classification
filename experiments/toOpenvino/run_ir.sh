#/bin/bash
ROOT=../..

IR_MODEL_FILE=${1-"output/qnet_exp2_ov-2021.4.582_sim-onnx/FP16/epoch_140.pth.xml"}
VIS_RATIO=${2-0}

tmp0=${IR_MODEL_FILE%/*}  # output/sddlite_exp2_4gpus_ov-2021.4.582/FP16
tmp1=${tmp0%/*}  # output/sddlite_exp2_4gpus_ov-2021.4.582
EXP_NAME=${tmp1##*/}  # sddlite_exp2_4gpus_ov-2021.4.582

# source /mnt/disk1/data_for_linjiaojiao/intel/openvino_2021.4.582/bin/setupvars.sh
export PYTHONPATH=${ROOT}:${PYTHONPATH}

if [[ ! -d "results" ]]; then
  mkdir results
fi


if [[ ! -d "logs" ]]; then
  mkdir logs
fi


JOB_NAME=test_ir_${EXP_NAME}-i8_uadetrac
OUTPUT_DIR=results/${JOB_NAME}
echo 'start: ' ${JOB_NAME}
echo output_dir: ${OUTPUT_DIR}

python -u ${ROOT}/tools/openvino_tools/run_qnet_ir.py \
    --ir_xml ${IR_MODEL_FILE} \
    --images_dir /mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/images \
    --list_file /mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_test.list \
    --py_pred_file /mnt/disk1/data_for_linjiaojiao/projects/open-mmlab/mmclassification/experiments/mv2/output/test_ua_detrac_mag_qnet_exp9/results.txt \
    -o ${OUTPUT_DIR} \
    -v ${VIS_RATIO} \
    --eval
    2>&1 | tee logs/${JOB_NAME}.log

# --list_file /mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_test_sparse.list \
