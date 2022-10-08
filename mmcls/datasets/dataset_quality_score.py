import os
from asyncio.log import logger
import numpy as np
from typing import Optional, Sequence, Union
import copy
import mmcv
from mmcls.core.evaluation.eval_metrics import mae_eval
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class QAImage(Dataset):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        test_mode (bool): In train mode or test mode. It's only a mark and
            won't be used in this class. Defaults to False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            If None, automatically inference from the specified path.
            Defaults to None.
    """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = ['quality_score']
    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):

        self.data_prefix = data_prefix
        self.pipeline = Compose(pipeline)
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.file_client_args = file_client_args
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load image paths and gt_labels."""
        if not isinstance(self.ann_file, list):
            self.ann_file = [self.ann_file]
        if not isinstance(self.data_prefix, list):
            self.data_prefix = [self.data_prefix]

        data_infos = []
        for data_prefix, ann_file in zip(self.data_prefix, self.ann_file):
            if isinstance(ann_file, str):
                lines = mmcv.list_from_file(
                    ann_file, file_client_args=self.file_client_args)
                for ln in lines:
                    # image-path quality-score
                    if ln.startswith('#'):
                        continue
                    tmp = ln.strip().split()
                    filename = tmp[0]
                    gt_score = float(tmp[1])
                    info = {
                        'img_prefix': data_prefix,
                        'img_info': {'filename': filename},
                        'gt_label': np.array(gt_score, dtype=np.float)
                        }
                    data_infos.append(info)
            else:
                raise TypeError('invalid ann_file: {}'.format(ann_file))
        print('load dataset done, {} items are loaded'.format(len(data_infos)))
        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def evaluate(self,
                 results,
                 metric='mae',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mae'. Options are 'mae'
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys: 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.

        Returns:
            dict: evaluation results
        """
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['mae']
        
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        
        eval_results = {}
        for cur_metric in metrics:
            if cur_metric == 'mae':
                mae_value = mae_eval(results, gt_labels)
                eval_results['mae'] = mae_value.item()
            else:
                raise ValueError('unknown metic: {}'.format(metric))

        return eval_results

    def _preds2list(self, results):
        """
        Associate results to each image.

        Return: 
        results = [[path, quality-score, gt-quality-score], ...]
        """
        assert len(results) == len(self), 'results mismatch with input-list: {} vs. {}'.format(len(results), len(self))

        res_list = []
        for idx in range(len(self)):
            result = results[idx]
            path = self.data_infos[idx]['img_info']['filename']
            gt_label = self.data_infos[idx]['gt_label']
            res_list.append([path, result, gt_label])
        return res_list

    def format_results(self, output, save_dir):
        res_list = self._preds2list(output)
        os.makedirs(save_dir, exist_ok=True)
        output_file = os.path.join(save_dir, 'results.txt')
        with open(output_file, 'w') as f:
            f.write('#path quality-score gt-quality-score\n')
            for res in res_list:
                path, result, gt_label = res
                f.write('{} {} {}\n'.format(path, result, gt_label))
        return output_file
