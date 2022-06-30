_base_ = [
    'SwinCascadeRCNN_config.py'
]

dataset_type = 'CocoDataset_5slices'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(type=dataset_type, ),
    val=dict(type=dataset_type, ),
    test=dict(type=dataset_type, )

)

model = dict(
    backbone=dict(
        type='SwinTransformer_5slices',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(
        type='FPN_proposed',
        in_channels=[96, 192, 384, 768],
        use_proposed_module=True,
        LN_MLP=True,
    ),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
