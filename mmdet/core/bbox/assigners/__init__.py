# Copyright (c) OpenMMLab. All rights reserved.
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from .center_region_assigner import CenterRegionAssigner
from .grid_assigner import GridAssigner
from .max_iou_assigner import MaxIoUAssigner
from .point_assigner import PointAssigner
from .sim_ota_assigner import SimOTAAssigner

__all__ = [
    'BaseAssigner', 'MaxIoUAssigner', 'AssignResult',
    'PointAssigner', 'CenterRegionAssigner', 'GridAssigner',
    'SimOTAAssigner','YOLOV5Assigner'
]
