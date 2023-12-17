_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

deepen_factor = 0.33
widen_factor = 0.25

model = dict(
    backbone=dict(
        type='CBYOLOv8CSPDarknet',
        out_indices=(1, 2, 3),
        cb_inplanes=[16, 32, 64, 128, 256],
        deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        type='CBYOLOv8PAFPN', 
        deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(
        type='CBYOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            widen_factor=widen_factor
            )
        ))
