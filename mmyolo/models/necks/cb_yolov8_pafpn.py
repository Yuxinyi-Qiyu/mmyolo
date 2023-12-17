# Copyright (c) OpenMMLab. All rights reserved.

from mmyolo.registry import MODELS
from .yolov8_pafpn import YOLOv8PAFPN
# from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN
    
@MODELS.register_module() 
class CBYOLOv8PAFPN(YOLOv8PAFPN):
    '''
    FPN with weight sharing
    which support mutliple outputs from cbnet
    '''
    #([1, 256, 80, 80],[1, 512, 40, 40],[1, 1024, 40, 40]),
    # ([1, 256, 80, 80],[1, 512, 40, 40],[1, 1024, 40, 40])
    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]
            
        if self.training:
            outs = []
            for x in inputs:
                out = super().forward(x)
                outs.append(out)
            return outs
        else:
            out = super().forward(inputs[-1])
            return out
        

if __name__ == "__main__":
    import torch

    # 给定的形状元组
    tuple00=torch.randn(torch.Size([1, 256, 80, 80]))
    tuple01=torch.randn(torch.Size([1, 512, 40, 40]))  
    tuple02=torch.randn(torch.Size([1, 1024, 20, 20]))

    tuple10=torch.randn(torch.Size([1, 256, 80, 80]))
    tuple11=torch.randn(torch.Size([1, 512, 40, 40]))  
    tuple12=torch.randn(torch.Size([1, 1024, 20, 20]))

    tuple0=(tuple00,tuple01,tuple02)
    tuple1=(tuple10,tuple11,tuple12)

    inputs=(tuple0, tuple1)

    model = CBYOLOv8PAFPN(in_channels=[256, 512, 1024],out_channels=[256, 512, 1024])
    level_outputs = model(inputs)
    for level_out in level_outputs:
        print(type(level_out))
        print(len(level_out))
        for t in level_out:
            print(type(t))
            print(t.shape)
