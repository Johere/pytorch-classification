# Vehicle Image Quality Assessment

The repository is for training vehicle image quality assessment models. Based on mmclassification framework.

## Training models

Training scripts are located under `experiments/*/train.sh`. 
Argments are specified using a config yaml file located in `configs/*`

## Evaluating models

Evaluation script is located under `experiments/*/test.sh`.
For convenience, config file used in testing stage will load from training-output-dir, for example, one experiment has been trained and saved in dir: output/base_exp0, test.sh try to load config file in `output/base_exp0/*.py`

## Exporting models to onnx and then to OpenVINO IR's

Check `experiments/toOpenvino/convert.sh` for the script.

After successfully converting, we can evaluate IR models using `experiments/toOpenvino/run_ir.sh` to check accuracy and recall for the resulted models. 

## Benchmark

### Model entry[1]: Mobilenet-v2, SmoothL1Loss regression

The training and testing scripts can be found in `experiments/mv2`
training scripts: `experiments/mv2/train.sh`
config file: `configs/mobilenet_v2/0.5mobilenet-v2_regression.py`
dataset: UA-DETRAC-fps5 + Boxcars

### Model entry[2]: 0.5 Mobilenet-v2, SmoothL1Loss regression
**also as release 0.0.1**

The training and testing scripts can be found in `experiments/mv2`
training scripts: `experiments/mv2/train.sh`
config file: `configs/mobilenet_v2/mobilenet-v2_regression.py`
dataset: UA-DETRAC-fps5 + Boxcars

### Model entry[3]: Mobilenet-v2-large, SmoothL1Loss regression

The training and testing scripts can be found in `experiments/mv2`
training scripts: `experiments/mv2/train.sh`
config file: `configs/mobilenet_v3/mobilenet-v3_large_regression.py`
dataset: UA-DETRAC-fps5 + Boxcars

### Model entry[4]: Mobilenet-v2-small, SmoothL1Loss regression

The training and testing scripts can be found in `experiments/mv2`
training scripts: `experiments/mv2/train.sh`
config file: `configs/mobilenet_v3/mobilenet-v3_small_regression.py`
dataset: UA-DETRAC-fps5 + Boxcars


**Results**
 |model|backbone|FLOPs|params|MAE@ua|MAE@boxcars|avg MAE|
 |------|----|----|----|----|----|----|
 |entry[1]|mv2|0.11G|688.96K|0.084|0.055|0.069|
 |entry[2]|0.5 mv2|0.32G|2.23M|0.086|0.055|0.070|
 |entry[3]|mv3-large|0.23G|4.2M|0.093|0.060|0.077|
 |entry[4]|mv3-small|0.06G|1.52M|0.094|0.062|0.078|