# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing.sharedctypes import Value
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

from .utils import weighted_loss


@weighted_loss
def ordinal_regression_loss(pred, target, value_segments=[0.5, 1.0]):
    """Ordinal Regression Loss
    https://raw.githubusercontent.com/ksnzh/DORN.pytorch/master/ordinal_regression_loss.py

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        value_segments (torch.Tensor): segments for oridnal regression

    Returns:
        torch.Tensor: Calculated loss
    """
    loss = 0
    import pdb; pdb.set_trace()
    for k in value_segments:
        # calculate the (2k, 2k+2) channel
        mask_lt = torch.gt(target, k)   # k-th: the current k is less than the gt label
        mask_gt = torch.le(target, k)   #

        target_lt = torch.ones_like(target)
        target_lt[mask_gt.data] = -100

        target_gt = torch.zeros_like(target)
        target_gt[mask_lt.data] = -100

        mask_lt = mask_lt.type(torch.cuda.FloatTensor).unsqueeze(1)
        count_lt = torch.sum(mask_lt)
        count_gt = mask_lt.size(0) * mask_lt.size(2) * mask_lt.size(3) - count_lt

        loss += F.cross_entropy(pred.narrow(1, 2*k, 2), target_lt) / (count_lt + 1)
        loss += F.cross_entropy(pred.narrow(1, 2*k, 2), target_gt) / (count_gt + 1)
    return loss


@LOSSES.register_module()
class OrdinalRegressionLoss(torch.nn.Module):
    def __init__(self, K, max_value=1.0, loss_weight=1.0):
        super(OrdinalRegressionLoss, self).__init__()

        self.CrossEntropy = torch.nn.CrossEntropyLoss(size_average=False, ignore_index=-100)
        self.loss_weight = loss_weight

        assert K > 1, 'invalid value K: {}'.format(K)
        step_val = max_value / K
        value_segments = [x * step_val for x in range(K)]
        self.value_segments = value_segments
        print('using ordinal regression loss with segments: {}'.format(self.value_segments))

    def forward(self,
                pred,
                target,
                weight=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction. (N * 2K)
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        num_pieces = len(self.value_segments) * 2
        if not pred.shape[1] // num_pieces * num_pieces == pred.shape[1]:
            import pdb; pdb.set_trace()
            raise ValueError('invalid prediction shape: {}, dim-1 should be divided by :{}'.format(pred.shape, num_pieces))

        loss = self.loss_weight * ordinal_regression_loss(
            pred,
            target,
            weight,
            value_segments=self.value_segments,
            **kwargs)
        return loss

