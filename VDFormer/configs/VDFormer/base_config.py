_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = '/data/huangjunjia/VDFormer/'

log_config = dict(
    interval=100, )

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(ann_file=data_root + 'infer.json',
               img_prefix=data_root + 'infer/', ),
    val=dict(ann_file=data_root + 'infer.json',
               img_prefix=data_root + 'infer/', ),
    test=dict(ann_file=data_root + 'infer.json',
               img_prefix=data_root + 'infer/', ),
)
