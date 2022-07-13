from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv
from mmdet.core.visualization import imshow_pred_det_bboxes, imshow_gt_det_bboxes
from pycocotools.coco import COCO
from mmcv import Config, DictAction
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.parallel import MMDataParallel
import torch
import numpy as np


def xywh2xyxy(bbox):
    _bbox = bbox
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] + _bbox[0],
        _bbox[3] + _bbox[1],
    ]


def main():
    img_root_path = "/data/huangjunjia/VDFormer/infer/"
    model_path = "/data/huangjunjia/VDFormer/"

    model_configs = ["SwinCascadeRCNN_VDFormer_config.py", ]
    model_checkpoints = ["VDFormer.pth", ]

    models = []
    dataloaders = []
    for i, (config, checkpoint) in enumerate(zip(model_configs, model_checkpoints)):
        config = './configs/VDFormer/' + config
        checkpoint = model_path + checkpoint
        cfg = Config.fromfile(config)
        print(cfg.get('test_cfg'))
        cfg.data.test.test_mode = True
        cfg.data.workers_per_gpu = 1
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        Checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in Checkpoint.get('meta', {}):
            model.CLASSES = Checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        model = MMDataParallel(model, device_ids=[4])

        models.append(model)
        dataloaders.append(data_loader)

    for i in range(len(model_configs)):
        models[i].eval()
        for j, data in enumerate(dataloaders[i]):
            with torch.no_grad():
                for j, data_batch in enumerate(data["img_metas"][0].data[0]):
                    slice_name = data_batch["ori_filename"]

                    img_path = img_root_path + slice_name
                    result = models[i](return_loss=False, rescale=True, **data)[0]
                    img = mmcv.imread(img_path)[:, :, 1]
                    img = img.copy()
                    img = imshow_pred_det_bboxes(img=img, result=result, score_thr=0.3,
                                               class_names=['tumours', ],
                                               win_name=slice_name)

                    mmcv.imwrite(img,
                                 "/data/huangjunjia/VDFormer/Infer_Output/" + slice_name[:-4] + ".png")


if __name__ == '__main__':
    main()
