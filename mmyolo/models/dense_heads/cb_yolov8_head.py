# Copyright (c) OpenMMLab. All rights reserved.


from mmyolo.registry import MODELS
from .yolov8_head import YOLOv8Head
from typing import Tuple, Union
from torch import Tensor

        
@MODELS.register_module()
class CBYOLOv8Head(YOLOv8Head):
     def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict], loss_weights=None) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        if isinstance(batch_data_samples, list):
            losses = super().loss(x, batch_data_samples)
        else:
           
            if not isinstance(x[0], (list, tuple)):
                x = [x]
                loss_weights = None
            elif loss_weights is None:
                loss_weights = [0.5] + [1]*(len(x)-1)  # Reference CBNet paper：[0.5, 1]
            
            def upd_loss(losses, idx, weight):
                new_losses = dict()
                for k,v in losses.items():
                    new_k = '{}{}'.format(k,idx)
                    if weight != 1 and 'loss' in k:
                        new_k = '{}_w{}'.format(new_k, weight)
                    if isinstance(v,list) or isinstance(v,tuple):
                        new_losses[new_k] = [i*weight for i in v]
                    else:new_losses[new_k] = v*weight
                return new_losses
            
            losses_out = dict()
            
            for i,x_split in enumerate(x):
                outs = self(x_split)
                # Fast version
                loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                                    batch_data_samples['img_metas'])
                losses = self.loss_by_feat(*loss_inputs)
                if len(x) > 1:
                    cb_losses = upd_loss(losses, idx=i, weight=loss_weights[i])
                losses_out.update(cb_losses)

            # for x_split in x:
            #     outs = self(x_split)
            #     # Fast version
            #     loss_inputs = outs + (batch_data_samples['bboxes_labels'],
            #                         batch_data_samples['img_metas'])
            #     losses = self.loss_by_feat(*loss_inputs)
            #     # 遍历当前循环得到的loss字典的键值对
            #     for key, value in losses.items():
            #         # 检查当前键是否已经在累加字典中
            #         if key in accumulated_losses:
            #             # 如果在，累加对应的值
            #             accumulated_losses[key] += 0.5*value
            #         else:
            #             # 如果不在，初始化累加字典中的值为当前值
            #             accumulated_losses[key] = value
            

        return losses_out
