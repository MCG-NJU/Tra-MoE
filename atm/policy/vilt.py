import numpy as np
from collections import deque
import robomimic.utils.tensor_utils as TensorUtils
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.transforms as T

from einops import rearrange, repeat

from atm.model import *
from atm.model.track_patch_embed import TrackPatchEmbed
from atm.policy.vilt_modules.transformer_modules import *
from atm.policy.vilt_modules.rgb_modules import *
from atm.policy.vilt_modules.language_modules import *
from atm.policy.vilt_modules.extra_state_modules import ExtraModalityTokens
from atm.policy.vilt_modules.policy_head import *
from atm.utils.flow_utils import ImageUnNormalize, sample_double_grid, tracks_to_video

import numpy as np
from PIL import Image
import time

###############################################################################
#
# A ViLT Policy
#
###############################################################################


# pos not use
class PositionalEncoding(nn.Module):
    def __init__(self, num_positions, d_model):
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Parameter(self._init_positional_encoding(num_positions, d_model),
                                               requires_grad=True)

    def _init_positional_encoding(self, num_positions, d_model):
        # init pos
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(num_positions, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, num_positions, d_model)
        return pe


# trajectory sequence length = 16
class LearnableParams(nn.Module):
    def __init__(self):
        super(LearnableParams, self).__init__()
        self.params = nn.Parameter(torch.full((16,), 0.5))

    def forward(self):
        return self.params



class BCViLTPolicy(nn.Module):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(
        self,
        obs_cfg,
        img_encoder_cfg,
        language_encoder_cfg,
        extra_state_encoder_cfg,
        track_cfg,
        spatial_transformer_cfg,
        temporal_transformer_cfg,
        policy_head_cfg,
        load_path=None,
    ):
        super().__init__()

        self._process_obs_shapes(**obs_cfg)

        # 1. encode image
        self._setup_image_encoder(**img_encoder_cfg)

        # 2. encode language (spatial)
        self.language_encoder_spatial = self._setup_language_encoder(
            output_size=self.spatial_embed_size, **language_encoder_cfg
        )

        # 3. Track Transformer module
        self._setup_track(**track_cfg)

        # 3. define spatial positional embeddings, modality embeddings, and spatial token for summary
        self._setup_spatial_positional_embeddings()

        # 4. define spatial transformer
        self._setup_spatial_transformer(**spatial_transformer_cfg)

        ### 5. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = self._setup_extra_state_encoder(
            extra_embedding_size=self.temporal_embed_size, **extra_state_encoder_cfg
        )

        # 6. encode language (temporal), this will also act as the TEMPORAL_TOKEN, i.e., CLS token for action prediction
        self.language_encoder_temporal = self._setup_language_encoder(
            output_size=self.temporal_embed_size, **language_encoder_cfg
        )

        # 7. define temporal transformer
        self._setup_temporal_transformer(**temporal_transformer_cfg)

        # 8. define policy head
        self._setup_policy_head(**policy_head_cfg)

    
        self.learnable_params = LearnableParams()


        if load_path is not None:

            self.load(load_path)
            self.track.load(f"{track_cfg.track_fn}/model_best.ckpt")
            

    def _process_obs_shapes(
        self, obs_shapes, num_views, extra_states, img_mean, img_std, max_seq_len
    ):
        self.img_normalizer = T.Normalize(img_mean, img_std)
        self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
        self.obs_shapes = obs_shapes
        self.policy_num_track_ts = obs_shapes["tracks"][0]
        self.policy_num_track_ids = obs_shapes["tracks"][1]
        self.num_views = num_views
        self.extra_state_keys = extra_states
        self.max_seq_len = max_seq_len
        # define buffer queue for encoded latent features
        self.latent_queue = deque(maxlen=max_seq_len)
        self.track_obs_queue = deque(maxlen=max_seq_len)

    def _setup_image_encoder(
        self, network_name, patch_size, embed_size, no_patch_embed_bias
    ):
        self.spatial_embed_size = embed_size
        self.image_encoders = []
        for _ in range(self.num_views):
            input_shape = self.obs_shapes["rgb"]
            self.image_encoders.append(
                eval(network_name)(
                    input_shape=input_shape,
                    patch_size=patch_size,
                    embed_size=self.spatial_embed_size,
                    no_patch_embed_bias=no_patch_embed_bias,
                )
            )
        self.image_encoders = nn.ModuleList(self.image_encoders)

        self.img_num_patches = sum([x.num_patches for x in self.image_encoders])

    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)

    def _setup_track(
        self, track_fn, policy_track_patch_size=None, use_zero_track=False
    ):
        """
        track_fn: path to the track model
        policy_track_patch_size: The patch size of TrackPatchEmbedding in the policy, if None, it will be assigned the same patch size as TrackTransformer by default
        use_zero_track: whether to zero out the tracks (ie use only the image)
        """
        track_cfg = OmegaConf.load(f"{track_fn}/config.yaml")
        self.use_zero_track = use_zero_track

        track_cfg.model_cfg.load_path = f"{track_fn}/model_best.ckpt"
        track_cls = eval(track_cfg.model_name)

        self.track = track_cls(**track_cfg.model_cfg)
        # freeze
        self.track.eval()
        for param in self.track.parameters():
            param.requires_grad = False

        self.num_track_ids = self.track.num_track_ids
        self.num_track_ts = self.track.num_track_ts
        self.policy_track_patch_size = (
            self.track.track_patch_size
            if policy_track_patch_size is None
            else policy_track_patch_size
        )
        if hasattr(self.track, "track_dim"):
            self.track_dim = self.track.track_dim
        else:
            self.track_dim = 2
        self.track_proj_encoder = TrackPatchEmbed(
            num_track_ts=self.policy_num_track_ts,
            num_track_ids=self.num_track_ids,
            patch_size=self.policy_track_patch_size,
            in_dim=self.track_dim + self.num_views,  # X, Y, one-hot view embedding
            embed_dim=self.spatial_embed_size,
        )

        self.track_id_embed_dim = 16
        self.num_track_patches_per_view = self.track_proj_encoder.num_patches_per_track
        self.num_track_patches = self.num_track_patches_per_view * self.num_views

    def _setup_spatial_positional_embeddings(self):
        # setup positional embeddings
        spatial_token = nn.Parameter(
            torch.randn(1, 1, self.spatial_embed_size)
        )  # SPATIAL_TOKEN
        img_patch_pos_embed = nn.Parameter(
            torch.randn(1, self.img_num_patches, self.spatial_embed_size)
        )
        track_patch_pos_embed = nn.Parameter(
            torch.randn(
                1,
                self.num_track_patches,
                self.spatial_embed_size - self.track_id_embed_dim,
            )
        )
        modality_embed = nn.Parameter(
            torch.randn(
                1,
                len(self.image_encoders) + self.num_views + 1,
                self.spatial_embed_size,
            )
        )  # IMG_PATCH_TOKENS + TRACK_PATCH_TOKENS + SENTENCE_TOKEN

        self.register_parameter("spatial_token", spatial_token)
        self.register_parameter("img_patch_pos_embed", img_patch_pos_embed)
        self.register_parameter("track_patch_pos_embed", track_patch_pos_embed)
        self.register_parameter("modality_embed", modality_embed)

        # for selecting modality embed
        modality_idx = []
        for i, encoder in enumerate(self.image_encoders):
            modality_idx += [i] * encoder.num_patches
        for i in range(self.num_views):
            modality_idx += (
                [modality_idx[-1] + 1]
                * self.num_track_ids
                * self.num_track_patches_per_view
            )  # for track embedding
        modality_idx += [modality_idx[-1] + 1]  # for sentence embedding
        self.modality_idx = torch.LongTensor(modality_idx)

    def _setup_extra_state_encoder(self, **extra_state_encoder_cfg):
        if len(self.extra_state_keys) == 0:
            return None
        else:
            return ExtraModalityTokens(
                use_joint=("joint_states" in self.extra_state_keys),
                use_gripper=("gripper_states" in self.extra_state_keys),
                use_ee=("ee_states" in self.extra_state_keys),
                **extra_state_encoder_cfg,
            )

    def _setup_spatial_transformer(
        self,
        num_layers,
        num_heads,
        head_output_size,
        mlp_hidden_size,
        dropout,
        spatial_downsample,
        spatial_downsample_embed_size,
        use_language_token=True,
    ):
        self.spatial_transformer = TransformerDecoder(
            input_size=self.spatial_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
        )

        if spatial_downsample:
            self.temporal_embed_size = spatial_downsample_embed_size
            self.spatial_downsample = nn.Linear(
                self.spatial_embed_size, self.temporal_embed_size
            )
        else:
            self.temporal_embed_size = self.spatial_embed_size
            self.spatial_downsample = nn.Identity()

        self.spatial_transformer_use_text = use_language_token

    def _setup_temporal_transformer(
        self,
        num_layers,
        num_heads,
        head_output_size,
        mlp_hidden_size,
        dropout,
        use_language_token=True,
    ):
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(
            input_size=self.temporal_embed_size
        )

        self.temporal_transformer = TransformerDecoder(
            input_size=self.temporal_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
        )
        self.temporal_transformer_use_text = use_language_token

        action_cls_token = nn.Parameter(torch.zeros(1, 1, self.temporal_embed_size))
        nn.init.normal_(action_cls_token, std=1e-6)
        self.register_parameter("action_cls_token", action_cls_token)

    def _setup_policy_head(self, network_name, **policy_head_kwargs):
        policy_head_kwargs["input_size"] = (
            self.temporal_embed_size
            + self.num_views
            * self.policy_num_track_ts
            * self.policy_num_track_ids
            * self.track_dim
        )

        action_shape = policy_head_kwargs["output_size"]
        self.act_shape = action_shape
        self.out_shape = np.prod(action_shape)
        policy_head_kwargs["output_size"] = self.out_shape
        self.policy_head = eval(network_name)(**policy_head_kwargs)

    @torch.no_grad()
    def preprocess(self, obs, track, action):
        """
        Preprocess observations, according to an observation dictionary.
        Return the feature and state.
        """
        b, v, t, c, h, w = obs.shape

        action = action.reshape(b, t, self.out_shape)

        obs = self._preprocess_rgb(obs)

        return obs, track, action

    @torch.no_grad()
    def _preprocess_rgb(self, rgb):
        rgb = self.img_normalizer(rgb / 255.0)
        return rgb

    def _get_view_one_hot(self, tr):
        """tr: b, v, t, tl, n, d -> (b, v, t), tl n, d + v"""
        b, v, t, tl, n, d = tr.shape

        #print("***************************************")
        #print(tr.shape)

        tr = rearrange(tr, "b v t tl n d -> (b t tl n) v d")
        one_hot = torch.eye(v, device=tr.device, dtype=tr.dtype)[None, :, :].repeat(
            tr.shape[0], 1, 1
        )
        tr_view = torch.cat([tr, one_hot], dim=-1)  # (b t tl n) v (d + v)
        tr_view = rearrange(
            tr_view,
            "(b t tl n) v c -> b v t tl n c",
            b=b,
            v=v,
            t=t,
            tl=tl,
            n=n,
            c=d + v,
        )
        return tr_view

    @torch.no_grad()
    def grid_points_3d(self, grid_points, h, w, depths, intrinsics):
        """
        Args:
            grid_points: b v t tl n 2
            h: int
            w: int
            depths: b v t tl h w
            intrinsics: b v 3 3
        """
        assert (
            h == depths.shape[-2] and w == depths.shape[-1]
        ), f"depths shape: {depths.shape}, h: {h}, w: {w}"
        grid_points = grid_points.clone()
        grid_points[..., 0] = grid_points[..., 0] * h
        grid_points[..., 1] = grid_points[..., 1] * w
        round_grid_points = grid_points.round().long()
        round_grid_points[..., 0] = torch.clamp(round_grid_points[..., 0], 0, h - 1)
        round_grid_points[..., 1] = torch.clamp(round_grid_points[..., 1], 0, w - 1)
        b, v, t, tl = round_grid_points.shape[:4]
        depths = rearrange(depths, "b v t tl h w -> (b v t tl) h w")
        round_grid_points = rearrange(
            round_grid_points, "b v t tl n c -> (b v t tl) n c", c=2
        )
        # depths torch.Size([2560, 128, 128]) round_grid_points torch.Size([40960, 32, 2])
        # nearest_depth: (b v t tl) n, index depths with round_grid_points[..., 0] and round_grid_points[..., 1]
        x_coords = round_grid_points[..., 0].long()  # shape: (b v t tl) n
        y_coords = round_grid_points[..., 1].long()  # shape: (b v t tl) n
        nearest_depth = depths[
            torch.arange(depths.shape[0]).unsqueeze(1), y_coords, x_coords
        ]  # shape: (b v t tl) n
        nearest_depth = rearrange(
            nearest_depth, "(b v t tl) n -> b v t tl n", b=b, v=v, t=t, tl=tl
        )

        points_3d = self.xy_to_points(grid_points, nearest_depth, intrinsics)
        return points_3d

    @torch.no_grad()
    def xy_to_points(self, grid_points, depths, intrinsics, extrinsic=None):
        """
        Transform 2D UV coordinates (grid points) to 3D coordinates.

        Parameters:
        - grid_points: Tensor of shape (b, v, t, tl, n, 2)
        - depths: Tensor of shape (b, v, t, tl, n)
        - intrinsics: Tensor of shape (b, v, 3, 3)
        - extrinsic: Tensor of shape (4, 4), default is identity matrix

        Returns:
        - points: Tensor of shape (b, v, t, tl, n, 3)
        """
        if extrinsic is None:
            extrinsic = torch.eye(4, device=grid_points.device)

        # Extract fx, fy, cx, cy from intrinsics
        fx = (
            intrinsics[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # shape: (b, v, 1, 1, 1)
        fy = (
            intrinsics[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # shape: (b, v, 1, 1, 1)
        cx = (
            intrinsics[:, :, 0, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # shape: (b, v, 1, 1, 1)
        cy = (
            intrinsics[:, :, 1, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # shape: (b, v, 1, 1, 1)

        # Compute points_x and points_y
        xmap = grid_points[..., 0]  # shape: (b, v, t, tl, n)
        ymap = grid_points[..., 1]  # shape: (b, v, t, tl, n)
        points_z = depths  # shape: (b, v, t, tl, n)

        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # Stack the points to get shape (b, v, t, tl, n, 3)
        points = torch.stack([points_x, points_y, points_z], dim=-1).float()

        # Add the homogeneous coordinate for transformation
        ones = torch.ones_like(points[..., :1])
        points_homogeneous = torch.cat(
            [points, ones], dim=-1
        )  # shape: (b, v, t, tl, n, 4)

        # Flatten the batch dimensions for matrix multiplication
        points_homogeneous_flat = points_homogeneous.view(
            -1, points_homogeneous.shape[-2], 4
        )  # shape: (b*v*t*tl, n, 4)

        # Apply the extrinsic transformation
        extrinsic = extrinsic.to(points.device)
        points_transformed = (
            points_homogeneous_flat @ extrinsic.T
        )  # shape: (b*v*t*tl, n, 4)

        # Reshape back to the original batch dimensions
        points_transformed = points_transformed.view(
            *points_homogeneous.shape
        )  # shape: (b, v, t, tl, n, 4)

        # Normalize the points by the last homogeneous coordinate
        points_transformed = points_transformed / points_transformed[..., -1:]

        # Remove the homogeneous coordinate
        points_3d = points_transformed[..., :-1]  # shape: (b, v, t, tl, n, 3)

        return points_3d

    def track_encode(self, track_obs, task_emb, extra_states):
        """
        Args:
            track_obs: b v t tt_fs c h w
            task_emb: b e
        Returns: b v t track_len n 2
        """
        assert self.num_track_ids == 32
        b, v, t, *_, h, w = track_obs.shape

        if self.use_zero_track:
            recon_tr = torch.zeros(
                (b, v, t, self.num_track_ts, self.num_track_ids, 2),
                device=track_obs.device,
                dtype=track_obs.dtype,
            )
        else:
            track_obs_to_pred = rearrange(
                track_obs, "b v t fs c h w -> (b v t) fs c h w"
            )

            grid_points = sample_double_grid(
                4, device=track_obs.device, dtype=track_obs.dtype
            )  # 32, 2
            grid_sampled_track = repeat(
                grid_points, "n d -> b v t tl n d", b=b, v=v, t=t, tl=self.num_track_ts
            )
            if self.track_dim == 3:
                with torch.no_grad():
                    depths = extra_states["depths"]  # b v t tl h w
                    intrinsics = extra_states["intrinsics"]  # b v 3 3
                    # track_obs (b, v, 10, 1, c, h, w)
                    # depths (b, v, 10, 1, h, w)
                    # grid_sampled_track torch.Size([b, v, 10, 16, 32, 2])
                    # intrinsics torch.Size([128, 2, 3, 3])
                    depths = depths[..., 0:1, :, :]
                    grid_sampled_track = grid_sampled_track[..., 0:1, :, :]
                    grid_sampled_track = self.grid_points_3d(
                        grid_sampled_track, h, w, depths, intrinsics
                    )  #  b v t tl n d, d=3
                    assert (
                        grid_sampled_track.shape[-1] == 3
                    ), f"grid_sampled_track shape: {grid_sampled_track.shape}"
                    grid_sampled_track = repeat(
                        grid_sampled_track,
                        "b v t 1 n d -> b v t tl n d",
                        tl=self.num_track_ts,
                    )
            grid_sampled_track = rearrange(
                grid_sampled_track, "b v t tl n d -> (b v t) tl n d"
            )

            expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
            expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
            with torch.no_grad():
                pred_tr, _ = self.track.reconstruct(
                    track_obs_to_pred, grid_sampled_track, expand_task_emb, p_img=0
                )  # (b v t) tl n d
                recon_tr = rearrange(
                    pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t
                )
                #print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

        recon_tr = recon_tr[
            :, :, :, : self.policy_num_track_ts, :, :
        ]  # truncate the track to a shorter one
        _recon_tr = recon_tr.clone()  # b v t tl n 2
        with torch.no_grad():
            tr_view = self._get_view_one_hot(recon_tr)  # b v t tl n c

        tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
        tr = self.track_proj_encoder(tr_view)  # (b v t) track_patch_num n d
        tr = rearrange(
            tr,
            "(b v t) pn n d -> (b t n) (v pn) d",
            b=b,
            v=v,
            t=t,
            n=self.num_track_ids,
        )  # (b t n) (v patch_num) d

        return tr, _recon_tr


    def compute_trajectory_lengths(self, trajectories):
        start_points = trajectories[:, :, 0, :, :]
        end_points = trajectories[:, :, -1, :, :]
        lengths = torch.norm(end_points - start_points, dim=-1)
        return lengths

    def compute_mean_distance(self, trajectories):
        mean_positions = trajectories.mean(dim=2)
        distances_to_mean = torch.norm(trajectories - mean_positions.unsqueeze(2), dim=-1)
        mean_distances = distances_to_mean.sum(dim=2)
        return mean_distances


    def spatial_encode(self, obs, track_obs, task_emb, extra_states, return_recon=False):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w, (0, 255)
            task_emb: b e
            extra_states: {k: b t n}
        Returns: out: (b t 2+num_extra c), recon_track: (b v t tl n 2)
        """

        # 2. encode task_emb
        text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
        B = obs.size(0)
        V = obs.size(1)
        T = obs.size(2)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c)

        # 3. encode track
        track_encoded, _recon_track = self.track_encode(track_obs, task_emb, extra_states)  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)

        obs = rearrange(obs, "b v t c h w -> (b v t) c h w")
        trajectories = rearrange(_recon_track, "b v t n p c -> (b v t) n p c")
        obs = obs*255.0

        new_channel = torch.zeros((B*V*T, 1, 128, 128), device=obs.device)
        obs = torch.cat([obs, new_channel], dim=1)

        #with torch.no_grad():

        _trajectories = trajectories.clone().detach()
        b, c, h, w = obs.shape
        _trajectories[:, :, :, 0] = _trajectories[:, :, :, 0] * w
        _trajectories[:, :, :, 1] = _trajectories[:, :, :, 1] * h

        traj_set = _trajectories.long() #bvt 32 16 2

        x, y = traj_set[:, :, :, 0], traj_set[:, :, :, 1] #bvt 32 16
        #mid_point = T // 2

        for i in range(0, 16):
            x_f, y_f = x[:, :, i:i+1], y[:, :, i:i+1] #bvt 32 1
            mask_f = (x_f >= 0) & (x_f < w) & (y_f >= 0) & (y_f < h)
            x_f, y_f = x_f[mask_f], y_f[mask_f]
            batch_indices_f = torch.arange(b, device=x_f.device).repeat_interleave(mask_f.sum(dim=[1, 2])).to(
                x_f.device)

            # print("****************************************************")
            # print(batch_indices_f.shape, y_f.shape, x_f.shape)
            # exit()

            # if (batch_indices_f % 2 == 0):
            #     obs[batch_indices_f, 3, y_f, x_f] = 255*self.learnable_params.params[i]
            # else:
            #     obs[batch_indices_f, 3, y_f, x_f] = 255*self.learnable_params2.params[i]

            #obs[batch_indices_f, 3, y_f, x_f] = 255*self.learnable_params.params[i]

            # view_1_indices = (batch_indices_f % 2 == 0)  # 假设视角1是每组视角的第一个
            # obs[batch_indices_f[view_1_indices], 3, y_f[view_1_indices], x_f[view_1_indices]] = 255 * self.learnable_params.params[i]
            # view_2_indices = (batch_indices_f % 2 == 1)  # 假设视角2是每组视角的第二个
            # obs[batch_indices_f[view_2_indices], 3, y_f[view_2_indices], x_f[view_2_indices]] = 255 * self.learnable_params2.params[i]

            obs[batch_indices_f, 3, y_f, x_f] = 255*self.learnable_params.params[i]

            
            
    
        obs = rearrange(obs, "(b v t) c h w -> b v t c h w", b=B, v=V, t=T)
        obs = obs / 255.0



        # 1. encode image
        img_encoded = []
        for view_idx in range(self.num_views):
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.image_encoders[view_idx]
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )  # (b, t, num_patches, c)

        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        B, T = img_encoded.shape[:2]





        # patch position embedding
        tr_feat, tr_id_emb = track_encoded[:, :, :-self.track_id_embed_dim], track_encoded[:, :, -self.track_id_embed_dim:]
        tr_feat += self.track_patch_pos_embed  # ((b t n), 2*patch_num, c)
        # track id embedding
        tr_id_emb[:, 1:, -self.track_id_embed_dim:] = tr_id_emb[:, :1, -self.track_id_embed_dim:]  # guarantee the permutation invariance
        track_encoded = torch.cat([tr_feat, tr_id_emb], dim=-1)
        track_encoded = rearrange(track_encoded, "(b t n) pn d -> b t (n pn) d", b=B, t=T)  # (b, t, 2*num_track*num_track_patch, c)

        #pdb.set_trace()

        # 3. concat img + track + text embs then add modality embeddings
        if self.spatial_transformer_use_text:
            img_track_text_encoded = torch.cat([img_encoded, track_encoded, text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 1, c)
            img_track_text_encoded += self.modality_embed[None, :, self.modality_idx, :]
        else:
            img_track_text_encoded = torch.cat([img_encoded, track_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch, c)
            img_track_text_encoded += self.modality_embed[None, :, self.modality_idx[:-1], :]

        # 4. add spatial token
        spatial_token = self.spatial_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c)
        encoded = torch.cat([spatial_token, img_track_text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)

        # 5. pass through transformer
        encoded = rearrange(encoded, "b t n c -> (b t) n c")  # (b*t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t
        out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')

        # 6. encode extra states
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')

        # 7. encode language, treat it as action token
        text_encoded_ = self.language_encoder_temporal(task_emb)  # (b, c')
        text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c')
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        if self.temporal_transformer_use_text:
            out_seq = [action_cls_token, text_encoded_, out]
        else:
            out_seq = [action_cls_token, out]

        if self.extra_encoder is not None:
            out_seq.append(extra)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')


        if return_recon:
            output = (output, _recon_track)
        return output
    

    def spatial_encode_atm(
        self, obs, track_obs, task_emb, extra_states, return_recon=False
    ):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w, (0, 255)
            task_emb: b e
            extra_states: {k: b t n}
        Returns: out: (b t 2+num_extra c), recon_track: (b v t tl n 2)
        """
        # 1. encode image
        img_encoded = []
        for view_idx in range(self.num_views):
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.image_encoders[view_idx]
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )  # (b, t, num_patches, c)

        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        B, T = img_encoded.shape[:2]

        # 2. encode task_emb
        text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (b, t, 1, c)

        # 3. encode track
        track_encoded, _recon_track = self.track_encode(
            track_obs, task_emb, extra_states
        )  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)


        # patch position embedding
        tr_feat, tr_id_emb = (
            track_encoded[:, :, : -self.track_id_embed_dim],
            track_encoded[:, :, -self.track_id_embed_dim :],
        )
        tr_feat += self.track_patch_pos_embed  # ((b t n), 2*patch_num, c)
        # track id embedding
        tr_id_emb[:, 1:, -self.track_id_embed_dim :] = tr_id_emb[
            :, :1, -self.track_id_embed_dim :
        ]  # guarantee the permutation invariance
        track_encoded = torch.cat([tr_feat, tr_id_emb], dim=-1)
        track_encoded = rearrange(
            track_encoded, "(b t n) pn d -> b t (n pn) d", b=B, t=T
        )  # (b, t, 2*num_track*num_track_patch, c)

        # 3. concat img + track + text embs then add modality embeddings
        if self.spatial_transformer_use_text:
            img_track_text_encoded = torch.cat(
                [img_encoded, track_encoded, text_encoded], -2
            )  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 1, c)
            img_track_text_encoded += self.modality_embed[None, :, self.modality_idx, :]
        else:
            img_track_text_encoded = torch.cat(
                [img_encoded, track_encoded], -2
            )  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch, c)
            img_track_text_encoded += self.modality_embed[
                None, :, self.modality_idx[:-1], :
            ]

        # 4. add spatial token
        spatial_token = self.spatial_token.unsqueeze(0).expand(
            B, T, -1, -1
        )  # (b, t, 1, c)
        encoded = torch.cat(
            [spatial_token, img_track_text_encoded], -2
        )  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)

        # 5. pass through transformer
        encoded = rearrange(
            encoded, "b t n c -> (b t) n c"
        )  # (b*t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t
        out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')

        # 6. encode extra states
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')

        # 7. encode language, treat it as action token
        text_encoded_ = self.language_encoder_temporal(task_emb)  # (b, c')
        text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (b, t, 1, c')
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(
            B, T, -1, -1
        )  # (b, t, 1, c')
        if self.temporal_transformer_use_text:
            out_seq = [action_cls_token, text_encoded_, out]
        else:
            out_seq = [action_cls_token, out]

        if self.extra_encoder is not None:
            out_seq.append(extra)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')

        if return_recon:
            output = (output, _recon_track)
        
        return output

    def temporal_encode(self, x):
        """
        Args:
            x: b, t, num_modality, c
        Returns:
        """
        pos_emb = self.temporal_position_encoding_fn(x)  # (t, c)
        x = x + pos_emb.unsqueeze(1)  # (b, t, 2+num_extra, c)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (b, t*num_modality, c)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # (b, t, num_modality, c)
        return x[:, :, 0]  # (b, t, c)

    def forward_(self, obs, track_obs, track, task_emb, extra_states):
        """
        Return feature and info.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, not used for training, only preserved for unified interface
            extra_states: {k: b t e}
        """
        x, recon_track = self.spatial_encode(
            obs, track_obs, task_emb, extra_states, return_recon=True
        )  # x: (b, t, 2+num_extra, c), recon_track: (b, v, t, tl, n, 2)
        x = self.temporal_encode(x)  # (b, t, c)
        # before rearrange x: torch.Size([128, 10, 64]) recon: torch.Size([128, 2, 10, 16, 32, 3])
        # before cat x: torch.Size([128, 10, 64]) recon: torch.Size([128, 10, 3072])


        #globalss = recon_track[:, :, :, 15:16, :, :] - recon_track[:, :, :, 0:1, :, :]
        #recon_track = torch.cat([recon_track, globalss], dim = 3)


        recon_track = rearrange(recon_track, "b v t tl n d -> b t (v tl n d)")
        x = torch.cat([x, recon_track], dim=-1)  # (b, t, c + v*tl*n*2)
        dist = self.policy_head(
            x
        )  # only use the current timestep feature to predict action
        return dist

    def forward(self, mode, *args, **kwargs):

        #mode = "vis"

        if mode == "vis":
            return self.forward_vis(*args, **kwargs)
        elif mode == "loss":

            #aaa = self.forward_vis(*args, **kwargs)

            return self.forward_loss(*args, **kwargs)
        elif mode == "act":
            return self.act(*args, **kwargs)

    def forward_loss(self, obs, track_obs, track, task_emb, extra_states, action):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, not used for training, only preserved for unified interface
            task_emb: b emb_size
            action: b t act_dim
        """

        #真机需要修改
        #action = (action/2048.0) -1.0

        obs, track, action = self.preprocess(obs, track, action)
        dist = self.forward_(obs, track_obs, track, task_emb, extra_states)
        loss = self.policy_head.loss_fn(dist, action, reduction="mean")

        ret_dict = {
            "bc_loss": loss.sum().item(),
        }

        if not self.policy_head.deterministic:
            # pseudo loss
            sampled_action = dist.sample().detach()
            mse_loss = F.mse_loss(sampled_action, action)
            ret_dict["pseudo_sampled_action_mse_loss"] = mse_loss.sum().item()

        ret_dict["loss"] = ret_dict["bc_loss"]
        return loss.sum(), ret_dict

    def forward_vis(self, obs, track_obs, track, task_emb, extra_states, action):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2
            task_emb: b emb_size
        Returns:
        """
        _, track, _ = self.preprocess(obs, track, action)
        track = track[
            :, :, 0, :, :, :
        ]  # (b, v, track_len, n, 2) use the track in the first timestep

        b, v, t, track_obs_t, c, h, w = track_obs.shape
        if t >= self.num_track_ts:
            track_obs = track_obs[:, :, : self.num_track_ts, ...]
            track = track[:, :, : self.num_track_ts, ...]
        else:
            last_obs = track_obs[:, :, -1:, ...]
            pad_obs = repeat(
                last_obs,
                "b v 1 track_obs_t c h w -> b v t track_obs_t c h w",
                t=self.num_track_ts - t,
            )
            track_obs = torch.cat([track_obs, pad_obs], dim=2)
            last_track = track[:, :, -1:, ...]
            pad_track = repeat(
                last_track, "b v 1 n d -> b v tl n d", tl=self.num_track_ts - t
            )
            track = torch.cat([track, pad_track], dim=2)

        grid_points = sample_double_grid(
            4, device=track_obs.device, dtype=track_obs.dtype
        )
        grid_track = repeat(
            grid_points, "n d -> b v tl n d", b=b, v=v, tl=self.num_track_ts
        )
        if self.track_dim == 3:
            with torch.no_grad():
                depths = extra_states["depths"]  # b v t tl h w
                intrinsics = extra_states["intrinsics"]  # b v 3 3
                grid_track = grid_track[..., None, 0:1, :, :].repeat(
                    1, 1, depths.shape[2], 1, 1, 1
                )
                grid_track = self.grid_points_3d(
                    grid_track, h, w, depths, intrinsics
                )  #  b v t tl n d, d=3
                assert (
                    grid_track.shape[-1] == 3
                ), f"grid_track shape: {grid_track.shape}"
                grid_track = repeat(
                    grid_track, "b v t 1 n d -> b v t tl n d", tl=self.num_track_ts
                )
                grid_track = grid_track[:, :, 0, :, :, :]

        all_ret_dict = {}
        for view in range(self.num_views):
            gt_track = track[:1, view]  # (1 tl n d)
            gt_track_vid = tracks_to_video(gt_track, img_size=h)
            combined_gt_track_vid = (
                (track_obs[:1, view, 0, :, ...] * 0.6 + gt_track_vid * 0.4)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )


            kwargs = dict()
            if self.track_dim == 3:
                kwargs["intrinsics"] = extra_states["intrinsics"][:1, view]
            #breakpoint()

            batch = {"vid": track_obs[:1, view, 0, :, ...],"track": grid_track[:1, view],"task_emb": task_emb[:1]}

            # _, ret_dict = self.track(
            #     "vis",
            #     track_obs[:1, view, 0, :, ...],
            #     grid_track[:1, view],
            #     task_emb[:1],
            #     p_img=0,
            #     **kwargs,
            # )
            #rebuttal
            _, ret_dict = self.track(
                "vis",
                batch,
            )

            ret_dict["combined_track_vid"] = np.concatenate(
                [combined_gt_track_vid, ret_dict["combined_track_vid"]], axis=-1
            )

            all_ret_dict = {
                k: all_ret_dict.get(k, []) + [v] for k, v in ret_dict.items()
            }

        for k, v in all_ret_dict.items():
            if k == "combined_image" or k == "combined_track_vid":
                all_ret_dict[k] = np.concatenate(
                    v, axis=-2
                )  # concat on the height dimension
            else:
                all_ret_dict[k] = np.mean(v)
        return None, all_ret_dict

    def act(self, obs, task_emb, extra_states):
        """
        Args:
            obs: (b, v, h, w, c)
            task_emb: (b, em_dim)
            extra_states: {k: (b, state_dim,)}
        """
        self.eval()
        B = obs.shape[0]

        # expand time dimenstion
        obs = rearrange(obs, "b v h w c -> b v 1 c h w").copy()

        if self.track_dim == 3:
            depths = extra_states.pop("depths")
            # print("depth max", depths.max())
            intrinsics = extra_states.pop("intrinsics")
            if len(intrinsics.shape) == 3:
                if B != 1:
                    print(
                        "intrinsics shape: ",
                        intrinsics.shape,
                        "b: ",
                        B,
                        "depths",
                        depths.shape,
                    )
                intrinsics = repeat(intrinsics, "v h w -> b v h w", b=B)
                assert intrinsics.shape[0] == B

        extra_states = {
            k: rearrange(v, "b e -> b 1 e") for k, v in extra_states.items()
        }

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        obs = torch.Tensor(obs).to(device=device, dtype=dtype)
        task_emb = torch.Tensor(task_emb).to(device=device, dtype=dtype)
        extra_states = {
            k: torch.Tensor(v).to(device=device, dtype=dtype)
            for k, v in extra_states.items()
        }

        if (obs.shape[-2] != self.obs_shapes["rgb"][-2]) or (
            obs.shape[-1] != self.obs_shapes["rgb"][-1]
        ):
            obs = rearrange(obs, "b v fs c h w -> (b v fs) c h w")
            obs = F.interpolate(
                obs,
                size=self.obs_shapes["rgb"][-2:],
                mode="bilinear",
                align_corners=False,
            )
            obs = rearrange(
                obs, "(b v fs) c h w -> b v fs c h w", b=B, v=self.num_views
            )

        if self.track_dim == 3:
            depths = rearrange(depths, "b v h w -> b v 1 1 h w")
            depths = torch.Tensor(depths).to(device=device, dtype=dtype)
            obs = torch.cat([obs, depths], dim=-3)

        while len(self.track_obs_queue) < self.max_seq_len:
            if self.track_dim == 3:
                # repeated
                self.track_obs_queue.append(obs.clone())
            else:
                self.track_obs_queue.append(torch.zeros_like(obs))
        self.track_obs_queue.append(obs.clone())
        track_obs = torch.cat(list(self.track_obs_queue), dim=2)  # b v fs c h w
        track_obs = rearrange(track_obs, "b v fs c h w -> b v 1 fs c h w")

        if self.track_dim == 3:
            depths = track_obs[..., -1, :, :]
            track_obs = track_obs[..., :-1, :, :]
            obs = obs[..., :-1, :, :]
            extra_states["depths"] = depths
            extra_states["intrinsics"] = torch.Tensor(intrinsics).to(
                device=device, dtype=dtype
            )


        obs = self._preprocess_rgb(obs)

        with torch.no_grad():
            x, rec_tracks = self.spatial_encode(
                obs,
                track_obs,
                task_emb=task_emb,
                extra_states=extra_states,
                return_recon=True,
            )  # x: (b, 1, 4, c), recon_track: (b, v, 1, tl, n, 2)
            self.latent_queue.append(x)
            x = torch.cat(list(self.latent_queue), dim=1)  # (b, t, 4, c)
            x = self.temporal_encode(x)  # (b, t, c)

            feat = torch.cat(
                [
                    x[:, -1],
                    rearrange(
                        rec_tracks[:, :, -1, :, :, :], "b v tl n d -> b (v tl n d)"
                    ),
                ],
                dim=-1,
            )

            action = self.policy_head.get_action(
                feat
            )  # only use the current timestep feature to predict action
            action = action.detach().cpu()  # (b, act_dim)

        action = action.reshape(-1, *self.act_shape)
        action = torch.clamp(action, -1, 1)
        if self.track_dim == 3:
            # rec_tracks: (10, 2, 1, 16, 32, 3)
            # intrinsics: (10, 2, 3, 3)
            v = rec_tracks.shape[1]
            projected_rec_tracks = []
            for view_idx in range(v):
                projected_rec_track = self.project_3d_tracks_to_2d(
                    rec_tracks[:, view_idx],
                    extra_states["intrinsics"][:, view_idx],
                )
                projected_rec_tracks.append(projected_rec_track)
            rec_tracks = torch.stack(projected_rec_tracks, dim=1)  # (10, 2, 1, 16, 32, 2)
            rec_tracks[..., 0] = rec_tracks[..., 0] / obs.shape[-1]
            rec_tracks[..., 1] = rec_tracks[..., 1] / obs.shape[-2]

        return action.float().cpu().numpy(), (
            None,
            rec_tracks[:, :, -1, :, :, :],
        )  # (b, *act_shape)

    def project_3d_tracks_to_2d(self, track, intrinsics):
        """
        Project 3D tracks to 2D using the camera intrinsics.

        Parameters:
        - track: Tensor of shape (b, ..., 3) representing the 3D tracks.
        - intrinsics: Tensor of shape (b, 3, 3) representing the camera intrinsics.

        Returns:
        - projected_points: Tensor of shape (b, ..., 2) representing the 2D projected points.
        """
        # Ensure track and intrinsics are on the same device
        device = track.device
        intrinsics = intrinsics.to(device)

        # Get the batch size
        b = track.shape[0]

        # Flatten the batch dimensions of track to shape (b, -1, 3)
        track_flat = track.view(
            b, -1, 3
        )  # shape: (b, N, 3), where N is the product of the remaining dimensions

        # Apply the intrinsic matrix to the 3D points
        projected_homogeneous = torch.bmm(
            track_flat, intrinsics.transpose(1, 2)
        )  # shape: (b, N, 3)

        # Normalize by the depth (Z-coordinate)
        projected_points = (
            projected_homogeneous[..., :2] / projected_homogeneous[..., 2:3]
        )  # shape: (b, N, 2)

        # Reshape back to the original batch dimensions
        projected_points = projected_points.view(
            *track.shape[:-1], 2
        )  # shape: (b, ..., 2)

        return projected_points

    def reset(self):
        self.latent_queue.clear()
        self.track_obs_queue.clear()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        #self.load_state_dict(torch.load(path, map_location="cpu"))

        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def train(self, mode=True):
        super().train(mode)
        self.track.eval()

    def eval(self):
        super().eval()
        self.track.eval()
