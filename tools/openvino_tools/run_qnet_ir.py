from operator import gt
import shutil
import cv2
import os
import argparse
import glob
import time
import numpy as np

# source ~/intel/openvino_2021.4.582/bin/setupvars.sh
from mmcls.utils.openvino_helper import get_IE_output

parser = argparse.ArgumentParser(description='run quality assessment [need IR model]',
                                 allow_abbrev=False)
parser.add_argument('--ir_xml', help='Path to qnet IR model dir, /path/to/xml_file')
parser.add_argument('--images_dir', type=str, help='image dir for test')
parser.add_argument('--list_file', type=str, help='image list_file for test')
parser.add_argument('--py_pred_file', type=str, default=None, help='pytorch prediction results for reference')
parser.add_argument('-o', '--output_dir', type=str, default='./results/IR-models/quality_scores',
                    help='Path to dump predict results')
parser.add_argument('-v', '--vis_ratio', type=float, default=0, help='visualization ratio')
parser.add_argument('--eval', action="store_true", help='eval with metrics: mae')
args = parser.parse_args()

"""
example:
python run_qnet_ir.py \
    --ir_xml '/mnt/disk1/data_for_linjiaojiao/projects/open-mmlab/mmclassification/experiments/toOpenvino/output/qnet_exp2_ov-2021.4.582_sim-onnx_backbone/FP16/epoch_140.pth.xml' \
    --images_dir /mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/images \
    --list_file /mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/quality_score/dataset-splits/magface_feature_norm_qa_mag0_trainval_normalize_test_sparse.list \
    --py_pred_file /mnt/disk1/data_for_linjiaojiao/projects/open-mmlab/mmclassification/experiments/res18/output/test_ua_detrac_mag_qnet_exp2/results.txt \
    -o ./results/IR-models/qnet_exp2_uadetrac \
    --eval
"""


def get_image(path, dst_size, color='bgr', flag='IR'):
    """
    :param color: bgr or rgb
    :param dst_size:
    :param path:
    :return:
    """
    if flag == 'IR':
        image = cv2.imread(path)
        return transform_fn(image, dst_size, color=color)
    else:
        raise ValueError('unknow flag:{}'.format(flag))


def transform_fn(image, dst_size, color='bgr'):
    if color is 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dst_size)
    x = image.transpose(2, 0, 1)
    return x


def glob_one_files(pattern):
    files = glob.glob(pattern)
    if len(files) == 1:
        return files[0]
    else:
        if len(files) == 0:
            print('IR model file not exists in {}'.format(pattern))
        else:
            print('multiple IR model files are found: {}'.format(files))
        raise ValueError


def inference_qnet(image_files, gt_dict=None, py_pred_dict=None, input_size=(224, 224)):
    """
    quality assessment net
    224 x 224. BGR

    """

    color_space = 'bgr'
    model_dir = os.path.dirname(args.ir_xml)

    if args.vis_ratio > 0:
        vis_dir = os.path.join(args.output_dir, 'visualize')
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    res_file = os.path.join(args.output_dir, 'results.txt')
    fout = open(res_file, 'w')
    fout.write('# path ir-quality-score py-quality-score\n')

    executable_model = None
    net = None
    global_start = time.time()

    # # debug
    # debug_cnt = 1000
    # image_files = image_files[:debug_cnt]

    pred_list = []
    gt_list = []
    py_pred_list = []
    for idx, image_path in enumerate(image_files):
        end = time.time()

        # image = cv2.imread(image_path)
        # h, w, c = image.shape

        rel_image_path = image_path.replace(args.images_dir, '')
        while rel_image_path[0] == '/':
            rel_image_path = rel_image_path[1:]

        # qnet inference
        ie_input = get_image(image_path, dst_size=input_size, color=color_space)

        executable_model, net, output = \
            get_IE_output(model_dir, ie_input, executable_model=executable_model, net=net)

        predictions = output['probs']
        # batch
        for b_ix in range(len(predictions)):
            quality_score = predictions[b_ix][0]
            pred_list.append(quality_score)

        if py_pred_dict is not None:
            py_pred = py_pred_dict[image_path]
            py_pred_list.append(py_pred)
            fout.write('{} {} {}\n'.format(rel_image_path, quality_score, py_pred))
        else:
            fout.write('{} {}\n'.format(rel_image_path, quality_score))
        
        if gt_dict is not None:
            gt_list.append(gt_dict[image_path])

        elapsed = time.time() - end
        if idx % 100 == 0:
            fout.flush()
            print('[{}/{}] time: {:.05f} s'.format(idx + 1, len(image_files), elapsed))
        
        if idx % 1000 == 0:
            if args.eval:
                assert len(gt_list) == len(pred_list)
                np_gt_list = np.array(gt_list)
                np_pred_list = np.array(pred_list)
                mae_val = np.mean(np.abs(np_pred_list - np_gt_list))
                print('mae: {:.05f}'.format(mae_val))

            if len(py_pred_list) > 0:
                assert len(py_pred_list) == len(pred_list)
                np_py_pred_list = np.array(py_pred_list)
                np_pred_list = np.array(pred_list)
                diff_preds = np.mean(np.abs(np_py_pred_list - np_pred_list))
                print('diff_preds: {:.05f}'.format(diff_preds))

    print('[{}] done. total time:{:.05f} s'.format(len(image_files), time.time() - global_start))

    fout.close()
    print('predicts saved: {}'.format(res_file))

    if args.eval:
        assert len(gt_list) == len(pred_list)
        gt_list = np.array(gt_list)
        pred_list = np.array(pred_list)
        mae_val = np.mean(np.abs(pred_list - gt_list))
        print('mae: {:.05f}'.format(mae_val))

    if len(py_pred_list) > 0:
        assert len(py_pred_list) == len(pred_list)
        py_pred_list = np.array(py_pred_list)
        pred_list = np.array(pred_list)
        diff_preds = np.mean(np.abs(py_pred_list - pred_list))
        print('diff_preds: {:.05f}'.format(diff_preds))

    if args.vis_ratio > 0:
        print('result images saved: {}'.format(vis_dir))


if __name__ == '__main__':
    
    with open(args.list_file, 'r') as f:
        lines = f.readlines()

    image_files = []
    gt_dict = dict()
    for ln in lines:
        if ln.startswith('#'):
            print('start reading list_file: {}'.format(ln))
            continue
        tmp = ln.strip().split()
        path = os.path.join(args.images_dir, tmp[0])
        image_files.append(path)
        if len(tmp) > 1:
            gt_dict[path] = float(tmp[1])
    if len(gt_dict.keys()) == 0:
        gt_dict = None
    
    py_pred_dict = None
    if args.py_pred_file is not None:
        py_pred_dict = dict()
        with open(args.py_pred_file, 'r') as f:
            ref_lines = f.readlines()
        for ln in ref_lines:
            if ln.startswith('#'):
                print('start reading list_file: {}'.format(ln))
                continue
            tmp = ln.strip().split()
            path = os.path.join(args.images_dir, tmp[0])
            py_pred_dict[path] = float(tmp[1])
        
    print('{} images to be inferenced.'.format(len(image_files)))

    inference_qnet(image_files, gt_dict=gt_dict, py_pred_dict=py_pred_dict, input_size=(224, 224))

    print('Done.')
