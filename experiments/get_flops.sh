#!/usr/bin/env bash
set -x

ROOT=..
export PYTHONPATH=${ROOT}:${PYTHONPATH}


CONFIG_FILE=${1-${ROOT}/configs/resnet/resnet18_regression.py}

INPUT_HEIGHT=${2-224}
INPUT_WIDTH=${3-${INPUT_HEIGHT}}

echo 'model complexity esitimate, config:' ${CONFIG_FILE}
python ${ROOT}/tools/analysis_tools/get_flops.py \
        ${CONFIG_FILE} \
        --shape ${INPUT_HEIGHT} ${INPUT_WIDTH}
        
# ```shell
# python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
# ```

# 用户将获得如下结果：

# ```
# ==============================
# Input shape: (3, 224, 224)
# Flops: 4.12 GFLOPs
# Params: 25.56 M
# ==============================
# ```