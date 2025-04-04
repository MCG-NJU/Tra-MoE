import numpy as np
from PIL import Image
from einops import repeat, rearrange
import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.obs_core import CropRandomizer

import torch.nn.functional as F


def get_dataloader(replay, mode, num_workers, batch_size):
    loader = DataLoader(
        replay,
        shuffle=(mode == "train"),
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader


def load_rgb(file_name):
    return np.array(Image.open(file_name))


class ImgTrackColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def forward(self, inputs):
        img = super().forward(inputs[0])
        return img, *inputs[1:]


class CropRandomizerReturnCoords(CropRandomizer):
    def _forward_in(self, inputs, return_crop_inds=False):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3  # must have at least (C, H, W) dimensions
        out, crop_inds = ObsUtils.sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        if return_crop_inds:
            return TensorUtils.join_dimensions(out, 0, 1), crop_inds
        else:
            # [B, N, ...] -> [B * N, ...]
            return TensorUtils.join_dimensions(out, 0, 1)


class ImgViewDiffTranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic. Take the same translation for a batch of images.
    """

    def __init__(
        self,
        input_shape,
        translation,
        augment_track=True,
    ):
        super().__init__()

        self.pad_translation = translation // 2
        pad_output_shape = (
            3,
            input_shape[0] + translation,
            input_shape[1] + translation,
        )

        self.crop_randomizer = CropRandomizerReturnCoords(
            input_shape=pad_output_shape,
            crop_height=input_shape[0],
            crop_width=input_shape[1],
        )
        self.augment_track = augment_track

    def forward(self, inputs):
        """
        Args:
            img: [b, t, C, H, W]
            tracks: [b, t, track_len, n, 2]
        """
        depths, intrinsics = None, None
        if len(inputs) == 2:
            img, tracks = inputs
        else:
            img, tracks, depths, intrinsics = inputs

    


        if depths is not None:
            # img: (b, 1, c, h, w), depths: (b, t, 1, h, w)
            # reshape and cat together:
            batch_size, temporal_len, img_c, img_h, img_w = img.shape
            img = rearrange(img, "b t c h w -> b (t c) h w")
            img_tc = img.shape[1]
            _, depth_temporal_len, depth_c, _, _ = depths.shape
            depths = rearrange(depths, "b t c h w -> b (t c) h w")
            depths_tc = depths.shape[1]
            img = torch.cat([img, depths], dim=1)
        else:
            batch_size, temporal_len, img_c, img_h, img_w = img.shape
            img = img.reshape(batch_size, temporal_len * img_c, img_h, img_w)




        out = F.pad(img, pad=(self.pad_translation,) * 4, mode="replicate")
        out, crop_inds = self.crop_randomizer._forward_in(
            out, return_crop_inds=True
        )  # crop_inds: (b, num_crops, 2), where we already set num_crops=1
        if depths is None:





            out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)
        else:
            img = out[:, :img_tc, :, :]
            depths = out[:, img_tc:, :, :]
            img = img.reshape(batch_size, temporal_len, img_c, img_h, img_w)
            depths = depths.reshape(batch_size, depth_temporal_len, depth_c, img_h, img_w)
            out = (img, depths)



        if self.augment_track:
            translate_h = (crop_inds[:, 0, 0] - self.pad_translation) / img_h  # (b,)
            translate_w = (crop_inds[:, 0, 1] - self.pad_translation) / img_w
            translate_h = repeat(translate_h, "b -> b 1 1 1")  # (b, 1, 1, 1)
            translate_w = repeat(translate_w, "b -> b 1 1 1")

            tracks[..., 0] += translate_h
            tracks[..., 1] += translate_w

        if intrinsics is not None:  # (b, 3, 3)
            intrinsics[:, 0, 2] += crop_inds[:, 0, 1] - self.pad_translation
            intrinsics[:, 1, 2] += crop_inds[:, 0, 0] - self.pad_translation
            return out, tracks, intrinsics

        return out, tracks
