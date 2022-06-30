from ..builder import BACKBONES
from .resnet import ResNet


@BACKBONES.register_module()
class ResNet_5slices(ResNet):
    def forward(self, x):
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4]).contiguous()
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


