import torch.nn.functional as F
from ..builder import BACKBONES
from .swin_transformer import SwinTransformer
import torch


@BACKBONES.register_module()
class SwinTransformer_frozen(SwinTransformer):
    def forward(self, x):
        with torch.no_grad():
            """Forward function."""
            # [2,5,3,800,800] -> [10,3,800,800]
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4]).contiguous()

            x = self.patch_embed(x)

            Wh, Ww = x.size(2), x.size(3)
            if self.ape:
                # interpolate the position embedding to the corresponding size
                absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
                x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            else:
                x = x.flatten(2).transpose(1, 2)
            x = self.pos_drop(x)

            outs = []
            for i in range(self.num_layers):
                layer = self.layers[i]
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x_out)

                    out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                    outs.append(out)

        return tuple(outs)


@BACKBONES.register_module()
class SwinTransformer_slices(SwinTransformer):
    def forward(self, x):
        """Forward function."""
        # [2,5,3,800,800] -> [10,3,800,800]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4]).contiguous()

        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)



@BACKBONES.register_module()
class SwinTransformer_5slices(SwinTransformer):
    def forward(self, x):
        """Forward function."""
        # [2,5,3,800,800] -> [10,3,800,800]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4]).contiguous()

        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)