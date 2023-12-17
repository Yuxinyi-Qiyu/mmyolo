# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import  build_norm_layer
from mmengine.model import constant_init



from mmyolo.registry import MODELS

from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from .csp_darknet import YOLOv8CSPDarknet
# from mmyolo.models.backbones.csp_darknet import YOLOv8CSPDarknet


class _CBNet(BaseModule):
    def _freeze_stages(self):
        for m in self.cb_modules:
            m._freeze_stages()
    
    def init_cb_weights(self):
        raise NotImplementedError

    def init_weights(self):
        self.init_cb_weights()
        for m in self.cb_modules:
            m.init_weights()

    def _get_cb_feats(self, feats, spatial_info):
        raise NotImplementedError

    def forward(self, x):
        outs_list = []
        for i, module in enumerate(self.cb_modules):  # 有两个backbone:assist and lead backbone
            if i == 0:
                pre_outs, spatial_info = module(x)  #  元组：包括每一层结束后的输出feature map, []
                # pre_outs:(x_stem,x_stage1,x_stage2,x_stage3,x_stage4)
                # spatial_info:(320,160,80,40)
            else:
                pre_outs, spatial_info = module(x, cb_feats, pre_outs)  # default:None,上一个backbone 的cb_feats
            # print(f"pre_outs:{len(pre_outs)}")
            # print(f"out_indices:{self.out_indices}")
            outs = [pre_outs[i+1] for i in self.out_indices]  # out_indices=(1, 2, 3), 不算stem层
            # outs：（x_stage1,x_stage2,x_stage3,x_stage4)
            outs_list.append(tuple(outs))
            
            if i < len(self.cb_modules)-1:  # i=0
                cb_feats = self._get_cb_feats(pre_outs, spatial_info)  
        return tuple(outs_list)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        for m in self.cb_modules:
            m.train(mode=mode)
        self._freeze_stages()
        for m in self.cb_linears.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, _BatchNorm):
                m.eval()
# backbone=dict(
#         frozen_stages=4,
#         type='CBYOLOv8CSPDarknet',
#         cb_del_stages=1,  # the number of stages in the backbone that are modified or affected by the "Composite Backbone" (CB) components
#         cb_inplanes=[64, 128, 256, 512, 1024], # the number of input channels for different stages of the backbone.
class _CBYOLOv8CSPDarknet(_CBNet):
    def __init__(self, net, cb_inplanes, cb_zero_init=True, cb_del_stages=0, **kwargs):
        super(_CBYOLOv8CSPDarknet, self).__init__()
        self.cb_zero_init = cb_zero_init  # True
        self.cb_del_stages = cb_del_stages  # 1
        # net = _ResNet/_Res2Net
        # cb_inplanes=[64, 128, 256, 512, 1024]
        # cb_del_stages=1
        self.cb_modules = nn.ModuleList()
        # two backbone: 0.assisting backbone 1.leadbackbone
        for cb_idx in range(2):  # 0/1:K=2
            cb_module = net(**kwargs)  # net:此时换为yolov8-darknet
            if cb_idx > 0:  # delete stem layer of backbone1(2/3/4...)
                cb_module.del_layers(self.cb_del_stages)  # 看看_ResNet怎么delete_layers的:delete stem/conv1
            self.cb_modules.append(cb_module)  # add：把delete好的网络模块聚一起
        self.out_indices = self.cb_modules[0].out_indices  # index:net[0].out_indices--assisting backbone.out_indices:(1, 2, 3, 4)
        # 五个block，每个block里有4个stage,stage-output(0\1\2\3)

        self.cb_linears = nn.ModuleList()
        # stage_blocks = (2, 2, 2, 2)：4个stage,每个stage里面都有2个BasicBlock模块
        self.num_layers = self.cb_modules[0].num_stages  # 4---stage_blocks
        norm_cfg = self.cb_modules[0].norm_cfg  # stage_blocks:_ResNet(assisting backbone)的norm_cfg是多少
        for i in range(self.num_layers):  # 4：0123 有4个stage
            linears = nn.ModuleList()
            if i >= self.cb_del_stages:  # 除了assisting backbone被删stem层后的其他stage：123
                jrange = 4 - i  # 3 2 1
                for j in range(jrange):
                    linears.append(  # 在每次内部循环迭代中，创建一个包含两个层的序列
                        nn.Sequential(
                            nn.Conv2d(cb_inplanes[i + j + 1], cb_inplanes[i], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                            build_norm_layer(norm_cfg, cb_inplanes[i])[1]
                        )
                    )
            
            self.cb_linears.append(linears)  # 每个stage都有一个linear,这个linear由每次内部循环迭代的网络序列组成
            # cb_inplanes=[64, 128, 256, 512, 1024]
            # i=0：j=4(0123):
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[1], cb_inplanes[0], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[0])[1]
                        # )
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[2], cb_inplanes[0], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[0])[1]
                        # )
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[3], cb_inplanes[0], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[0])[1]
                        # )
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[4], cb_inplanes[0], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[0])[1]
                        # )
            # i=1, j=3(012): 
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[2], cb_inplanes[1], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[1])[1]
                        # )
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[3], cb_inplanes[1], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[1])[1]
                        # )
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[4], cb_inplanes[1], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[1])[1]
                        # )
            # i=2, j=2(01)
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[3], cb_inplanes[2], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[2])[1]
                        # )
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[4], cb_inplanes[2], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[2])[1]
                        # )
            # i=3, j=1(0)
            # nn.Sequential(
                        #     nn.Conv2d(cb_inplanes[4], cb_inplanes[3], 1, bias=False),  # 输出通道限制为了层1的固定输入维度
                        #     build_norm_layer(norm_cfg, cb_inplanes[3])[1]
                        # )
            # ModuleList(

            #   (0): ModuleList(
            #     (0): Sequential(
            #       (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #     (1): Sequential(
            #       (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #     (2): Sequential(
            #       (0): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #     (3): Sequential(
            #       (0): Conv2d(1024, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #   )

            #   (1): ModuleList(
            #     (0): Sequential(
            #       (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #     (1): Sequential(
            #       (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #     (2): Sequential(
            #       (0): Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #   )

            #   (2): ModuleList(
            #     (0): Sequential(
            #       (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #     (1): Sequential(
            #       (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #   )

            #   (3): ModuleList(
            #     (0): Sequential(
            #       (0): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            #   )
            # )

            # len(cb_linears)=4

    def init_cb_weights(self):
        if self.cb_zero_init:
            for ls in self.cb_linears:
                for m in ls:
                    if isinstance(m, nn.Sequential):
                        constant_init(m[-1], 0)
                    else:
                        constant_init(m, 0)
########################################################！！！！！！！！！！！
    # outs:(x_stem,x_stage1,x_stage2,x_stage3,x_stage4)
    # spatial_info:(320,160,80,40)
    def _get_cb_feats(self, feats, spatial_info):
        cb_feats = []
        for i in range(self.num_layers):  # 0123
            if i >= self.cb_del_stages:  # 0123
                h, w = spatial_info[i]  # 从上一层来的输出维度（320，320）（160，160）（80，80）（40，40）
                feeds = []
                jrange = 4 - i  # j: 0/1/2
                for j in range(jrange):  # 如图cbnetv2.png
                    tmp = self.cb_linears[i][j](feats[j + i + 1])  # feats2/3/4 ,寻找每个层的每个内部linear模块，把输入传进去--调整通道数为层1所接收的一般通道数
                    tmp = F.interpolate(tmp, size=(h, w), mode='nearest')  # 对输入进行插值操作，调整空间维度到这层（1）的feature维度
                    feeds.append(tmp)
                feed = torch.sum(torch.stack(feeds, dim=-1), dim=-1)  # 将多个特征图（或张量）合并为一个，并且在这个过程中进行了求和操作
            else:
                feed = 0
            cb_feats.append(feed)

        return cb_feats
    # i=0,j=0123:
    # h, w = spatial_info[0]--（320，320）
    # tmp = self.cb_linears[0][0](feats[1])input:x_stage1
    # tmp = self.cb_linears[0][1](feats[2])
    # tmp = self.cb_linears[0][2](feats[3])
    # tmp = self.cb_linears[0][3](feats[4])
    # i=1,j=012: 
    # h, w = spatial_info[1]--（160，160）
    # tmp = self.cb_linears[1][0](feats[2])
    # tmp = self.cb_linears[1][1](feats[3])
    # tmp = self.cb_linears[1][2](feats[4])
    # i=2,j=01:
    # h, w = spatial_info[2]
    # tmp = self.cb_linears[2][0](feats[3])
    # tmp = self.cb_linears[2][1](feats[4])
    # i=3,j=0:
    # h, w = spatial_info[3]
    # tmp = self.cb_linears[3][0](feats[4])



class _CBSubnet(BaseModule):  # pre_outs, spatial_info = module(x, cb_feats, pre_outs)
    def _freeze_stages(self):
        
        for i in range(1, self.frozen_stages + 1):
            if not hasattr(self, self.layers[i]):
                continue
            m = getattr(self, self.layers[i])
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


    def del_layers(self, del_stages):
        self.del_stages = del_stages  # = 1
          
        for i in range(self.del_stages + 1):  # 1
            delattr(self, self.layers[i])

    def forward(self, x, cb_feats=None, pre_outs=None):
        """Forward function."""
        spatial_info = []
        outs = []

        if hasattr(self, 'stem'):  # hasattr：用于检查一个对象（在这里是 self，即类的实例）是否具有指定的属性或方法。
            # 在你的代码中，这行代码用于检查当前类的实例是否具有名为 stem 的属性。
            # Stem模块是整个神经网络的入口，它负责对输入数据进行一些初步的处理:stem部分其实就是多次卷积＋２次pooling
            # maxpool:减小特征图的尺寸
            x = self.stem(x)
            # x = self.maxpool(x)
      
        else:
            x = pre_outs[0]  # 共享backbone1的stem
        outs.append(x)
        
        ###############################上面应该是准备加和之前的操作###################################
        # 对于backbone1:cb_feats=None
        # 0:layers=['stem','stage1','stage2']
        # 1:layers=['stage1','stage2']
        for i, layer_name in enumerate(self.layers):  
            if layer_name == 'stem': 
                continue

            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                spatial_info.append(x.shape[2:])  # spatial dimensions空间维度--w/h
                if cb_feats is not None:  # composite:相加(外来的feature map)
                    x = x + cb_feats[i-1]
                x = layer(x)  # 这层和上一个backbone融合好以后一起进入下一层
            else:
                x = pre_outs[i]  
            outs.append(x)  # 每一层结束后的输出结果（最终被转换为元组）
        return tuple(outs), spatial_info  # 每一层结束后的输出结果（最终被转换为元组）/未相融前feature map的spatial dimensions空间维度--w/h
        # outs:(x_stem,x_stage1,x_stage2,x_stage3,x_stage4)
        # spatial_info:(320,160,80,40)

        # outs:(x_stem,x_stage1,x_stage2,x_stage3,x_stage4)
        # spatial_info:(320,160,80,40)


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

class _YOLOv8CSPDarknet(_CBSubnet, YOLOv8CSPDarknet):
    def __init__(self, **kwargs):
        # 将 kwargs 中包含的任意关键字参数传递给 ResNet 的构造函数，以确保在子类 _ResNet 中可以正常地初始化 ResNet 的实例。
        _CBSubnet.__init__(self)  # 不接受外来参数
        YOLOv8CSPDarknet.__init__(self, **kwargs)  # ResNet is designed to accept and use **kwargs for its initialization.

@MODELS.register_module()
class CBYOLOv8CSPDarknet(_CBYOLOv8CSPDarknet):
    def __init__(self, **kwargs):
        super().__init__(net=_YOLOv8CSPDarknet, **kwargs)

if __name__ == "__main__":
    model = CBYOLOv8CSPDarknet(cb_inplanes=[64, 128, 256, 512, 1024], out_indices=(1, 2, 3))
    model.eval()
    inputs = torch.rand(1, 3, 640, 640)
    level_outputs = model(inputs)
    print(type(level_outputs))
    print(len(level_outputs))
    # ((torch.Size([1, 256, 80, 80]),torch.Size([1, 512, 40, 40]),torch.Size([1, 1024, 20, 20])),(torch.Size([1, 256, 80, 80]),torch.Size([1, 512, 40, 40]),torch.Size([1, 1024, 20, 20])))
    for level_out in level_outputs:
        print(type(level_out))
        print(len(level_out))
        for t in level_out:
            print(type(t))
            print(t.shape)


