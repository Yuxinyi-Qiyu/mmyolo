_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

model = dict(

    backbone=dict(
        type='CBYOLOv8CSPDarknet',
        out_indices=(1, 2, 3),
        cb_inplanes=[32, 64, 128, 256, 512]),
    neck=dict(
        type='CBYOLOv8PAFPN'),
    bbox_head=dict(
        type='CBYOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            )))
