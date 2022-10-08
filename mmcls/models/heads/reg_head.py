# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

from mmcls.models.losses import MAEAccuracy
from ..builder import HEADS, build_loss
from ..utils import is_tracing
from .base_head import BaseHead


@HEADS.register_module()
class RegHead(BaseHead):
    """regression head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
        use_sigmoid (bool): Whether the prediction uses sigmoid. Defaults to False.
    """

    def __init__(self,
                 loss=dict(type='SmoothL1Loss', loss_weight=1.0),
                 cal_acc=False,
                 init_cfg=None,
                 use_sigmoid=False):
        super(RegHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)
        self.compute_loss = build_loss(loss)
        self.compute_accuracy = MAEAccuracy()
        self.cal_acc = cal_acc
        self.use_sigmoid = use_sigmoid

    def loss(self, cls_score, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            losses['accuracy'] = {
                f'mae': acc
            }
        losses['loss'] = loss
        return losses

    def forward_train(self, cls_score, gt_label, **kwargs):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
            
        if self.use_sigmoid:
            pred = cls_score.sigmoid()
        else:
            pred = cls_score
        losses = self.loss(pred, gt_label, **kwargs)
        return losses

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]

        warnings.warn(
            'The input of ClsHead should be already logits. '
            'Please modify the backbone if you want to get pre-logits feature.'
        )
        return x

    def simple_test(self, cls_score, post_process=True):
        """Inference without augmentation.

        Args:
            cls_score (tuple[Tensor]): The input classification score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]

        if self.use_sigmoid:
            pred = cls_score.sigmoid()
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
