from .builder import DATASETS
from .coco import CocoDataset
import os.path as osp
from .pipelines import Compose
import torch


@DATASETS.register_module()
class CocoDataset_7slices(CocoDataset):
    CLASSES = ('tumour',)

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)
        self.data_infos_all = self.data_infos

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        result_data = []
        img_id = self.data_infos[idx]['id']
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=self.data_infos[idx], ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)

        for i in [-3, -2, -1, 1, 2, 3]:
            img_idx = img_id + i
            if img_idx < 0 or img_idx >= len(self.data_infos_all):
                data = torch.zeros(results['img'].data.shape)
                result_data.append(data)
                continue
            img_info = self.data_infos_all[img_idx]
            if img_info['file_name'][:12] != results['img_metas'].data['ori_filename'][:12]:
                data = torch.zeros(results['img'].data.shape)
                result_data.append(data)
                continue

            data_tmp = dict(img_info=img_info, ann_info=ann_info)
            if self.proposals is not None:
                data_tmp['proposals'] = self.proposals[idx]
            self.pre_pipeline(data_tmp)
            result_data.append(self.pipeline(data_tmp)['img'].data)

        results['img']._data = torch.stack(
            (result_data[0], result_data[1], result_data[2], results['img'].data, result_data[3], result_data[4],
             result_data[5],), 0).contiguous()
        return results

    def prepare_test_img(self, idx):
        result_data = []
        img_id = self.data_infos[idx]['id']
        results = dict(img_info=self.data_infos[idx])
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)

        for i in [-3, -2, -1, 1, 2, 3]:
            img_idx = img_id + i
            if img_idx < 0 or img_idx >= len(self.data_infos_all):
                data = torch.zeros(results['img'][0].data.shape)
                result_data.append(data)
                continue
            img_info = self.data_infos_all[img_idx]
            if img_info['file_name'][:12] != results['img_metas'][0].data['ori_filename'][:12]:
                data = torch.zeros(results['img'][0].data.shape)
                result_data.append(data)
                continue

            data_tmp = dict(img_info=img_info)
            if self.proposals is not None:
                data_tmp['proposals'] = self.proposals[idx]
            self.pre_pipeline(data_tmp)
            result_data.append(self.pipeline(data_tmp)['img'][0].data)

        results['img'][0] = torch.stack(
            (result_data[0], result_data[1], result_data[2], results['img'][0].data, result_data[3], result_data[4],
             result_data[5]), 0).contiguous()
        return results
