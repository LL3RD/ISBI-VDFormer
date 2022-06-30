from ..builder import NECKS
from .fpn import FPN
import torch.nn as nn
from mmcv.runner import auto_fp16
import torch.nn.functional as F
from .swin_fusion_layer_proposed import SwinFusionLayer
import torch
from ..utils import A3DConv
import time

@NECKS.register_module()
class FPN_proposed(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 use_proposed_module=False,
                 LN_MLP=True,
                 slices=5
                 ):
        super(FPN_proposed, self).__init__(in_channels,
                                           out_channels,
                                           num_outs,
                                           start_level,
                                           end_level,
                                           add_extra_convs,
                                           extra_convs_on_inputs,
                                           relu_before_extra_convs,
                                           no_norm_on_lateral,
                                           conv_cfg,
                                           norm_cfg,
                                           act_cfg,
                                           upsample_cfg, )

        self.use_proposed_module = use_proposed_module
        self.slices = slices
        if self.use_proposed_module:
            self.swin_fusion_layer = nn.ModuleList([
                SwinFusionLayer(
                    dim=256,
                    LN_MLP=LN_MLP,
                    slices=slices
                ) for i in range(5)  # FPN stages
            ])

    @auto_fp16()
    def forward(self, inputs):
        # inputs: ([2, 256, 200, 200], [2, 512, 100, 100], [2, 1024, 50, 50], [2, 2048, 25, 25]

        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        if self.use_proposed_module:
            for i, out in enumerate(outs):
                # print(out.shape)
                # if i == 1:
                outs[i] = self.swin_fusion_layer[i](out)
                # else:
                #     outs[i] = out.view(-1, 5, out.size(1), out.size(2), out.size(3))[:, 2, :, :, :].contiguous()

        print(time.time())
        return tuple(outs)  # B, C, H, W

    def inter_fusion(self, out, i):
        out = out.reshape(-1, self.slices * out.shape[1], out.shape[2], out.shape[3])
        out = self.inter_layer_fusion[i](out)
        return out
