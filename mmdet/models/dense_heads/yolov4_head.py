import math
from pickle import TRUE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,normal_init,
                      constant_init, is_norm, bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import (BaseModule, force_fp32)

from mmdet.core import (build_prior_generator, build_bbox_coder, 
                        bbox_xyxy_to_cxcywh, build_assigner,
                         build_sampler, multi_apply, images_to_levels,
                         multiclass_nms,reduce_mean)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin



@HEADS.register_module()
class YOLOV4Head(BaseDenseHead, BBoxTestMixin):
    """Head head used in YOLOV4.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    # From left to right:
    # use_auxhead
    arch_settings = {
        'L': False,
        'M': True,
        'S': True,
        'X': True
    }

    def __init__(self,
                 num_classes,
                 in_channels,
                 arch='L',
                 anchor_generator=dict(
                     type='YOLOV5AnchorGenerator',
                     base_sizes=[[(10, 13), (16, 30), (33, 23)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(116, 90), (156, 198), (373, 326)]],
                     strides=[8, 16, 32]),
                 bbox_coder=dict(type='YOLOV5BBoxCoder'),
                 strides=[8, 16, 32],
                 one_hot_smoother=0.,
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):

        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True
        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)

        self.prior_generator = build_prior_generator(anchor_generator)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(self.train_cfg.sampler)
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        assert len(
            self.prior_generator.num_base_priors) == len(strides)
        self.fp16_enabled = False
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes


    def _init_layers(self):
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                self.num_base_priors * self.num_attrib, 1)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

        # Use prior in model initialization to improve stability
        for conv_pred, stride in zip(self.convs_pred, self.strides):
            bias = conv_pred.bias.reshape(self.num_base_priors, -1)
            # init objectness with prior of 8 objects per feature map
            # refer to https://github.com/ultralytics/yolov3
            nn.init.constant_(bias.data[:, 4],
                              bias_init_with_prob(8 / (640 / stride)**2))
            nn.init.constant_(bias.data[:, 5:], bias_init_with_prob(0.01))

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)
        return tuple(pred_maps),

    @force_fp32(apply_to=('pred_maps'))
    def get_bboxes(self,
                   pred_maps,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(pred_maps) == self.num_levels
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = np.array(
            [img_meta['scale_factor'] for img_meta in img_metas])

        num_imgs = len(img_metas)
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=pred_maps[0].dtype,
            device=pred_maps[0].device,
            with_stride=True)

        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_attrib)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(
                pred.new_tensor(stride).expand(pred.size(1)))

        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()

        flatten_priors = torch.cat(mlvl_priors)
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = self.bbox_coder.decode(flatten_priors,
                                                flatten_bbox_preds,
                                                flatten_strides.unsqueeze(-1))
        if with_nms and (flatten_objectness.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            flatten_bboxes /= flatten_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        padding = flatten_bboxes.new_zeros(num_imgs, flatten_bboxes.shape[1], 1)
        flatten_cls_scores = torch.cat([flatten_cls_scores, padding], dim=-1)

        result_list = []
        for (bboxes, scores, objectness) in zip(flatten_bboxes,
                                                flatten_cls_scores,
                                                flatten_objectness):
            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            if conf_thr > 0:
                conf_inds = objectness >= conf_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=objectness)
            result_list.append(tuple([det_bboxes, det_labels]))
        return result_list

    @force_fp32(apply_to=('pred_maps'))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]

        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=pred_maps[0].dtype,
            device=pred_maps[0].device)

        anchor_list = [mlvl_priors for _ in range(num_imgs)]
        responsible_flag_list = []
        for img_id in range(len(img_metas)):
            responsible_flag_list.append(
                self.prior_generator.responsible_flags(featmap_sizes,
                                                       gt_bboxes[img_id],
                                                       pred_maps[0].device))
        valid_flag_list = []
        for img_id in range(len(img_metas)):
            valid_flag_list.append(
                self.prior_generator.valid_flags(featmap_sizes,
                                                       gt_bboxes[img_id],
                                                       pred_maps[0].device))                                                

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, valid_flag_list, responsible_flag_list, gt_bboxes, gt_labels)

        losses_cls, losses_obj, losses_bbox = multi_apply(
            self.loss_single, pred_maps, target_maps_list, neg_maps_list)

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_obj=losses_obj)

    def loss_single(self, pred_map, target_map, neg_map):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask
        pos_mask = pos_mask.unsqueeze(dim=-1)
        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        pred_bbox = pred_map[..., :4]
        pred_obj = pred_map[..., 4]
        pred_label = pred_map[..., 5:]

        target_bbox = target_map[..., :4]
        target_obj = target_map[..., 4]
        target_label = target_map[..., 5:]

        loss_cls = self.loss_cls(
            pred_label, target_label, weight=pos_mask)
        loss_obj = self.loss_obj(
            pred_obj, target_obj, weight=pos_and_neg_mask)
        loss_bbox = self.loss_bbox(pred_bbox, target_bbox, weight=pos_mask.expand(pos_mask.size(0),pos_mask.size(1),4))

        return loss_cls, loss_obj, loss_bbox

    def get_targets(self, anchor_list, valid_flag_list, responsible_flag_list, gt_bboxes_list,
                    gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(anchor_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              valid_flag_list, responsible_flag_list,
                              gt_bboxes_list, gt_labels_list)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    @torch.no_grad()
    def _get_targets_single(self, anchors, valid_flags, responsible_flags, gt_bboxes,
                            gt_labels):
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """

        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(
                torch.tensor(self.strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)
        concat_responsible_flags = torch.cat(responsible_flags,dim=1)
        concat_valid_flags = torch.cat(valid_flags)

        anchor_strides = torch.cat(anchor_strides)
        # assert len(anchor_strides) == len(concat_anchors) == \
            #    len(concat_responsible_flags)
        assign_result = self.assigner.assign(concat_anchors,
                                             concat_valid_flags,
                                             concat_responsible_flags,
                                             gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, concat_anchors,
                                              gt_bboxes)

        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib)
        target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
            sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes,
            anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 4] = 1

        gt_labels_one_hot = F.one_hot(
            gt_labels, num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:  # label smooth
            gt_labels_one_hot = gt_labels_one_hot * (
                1 - self.one_hot_smoother
            ) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[
            sampling_result.pos_assigned_gt_inds.long()]

        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1

        return target_map, neg_map

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
