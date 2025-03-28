import hydra
import math
from glob import glob
import pandas as pd
import torch

import datetime
import torch
torch.distributed.constants._DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=5000)

import torch.distributed as dist
import lightning
from lightning.fabric import Fabric
import gc

import os

from einops import rearrange
from omegaconf import DictConfig

from atm.policy import *
from atm.utils.train_utils import setup_optimizer
from atm.utils.env_utils import build_env
from engine.utils import rollout, merge_results
from atm.utils.video_utils import make_grid_video_from_numpy

from engine.utils import rollout

def get_ckp_name(file_path):
    return os.path.basename(file_path).split('.ckpt')[0].split('_')[-1]


def sort_ckp_paths(file_list, reverse=False):
    required_names = ["final", "best"]

    epoch2path = []
    name2path = {}
    for path in file_list:
        name = get_ckp_name(path)
        if name.isdigit():
            # epoch number checkpoint
            epoch = int(name)
            epoch2path.append((epoch, path))
        else:
            # final / best checkpoint
            name2path[name] = path

    sorted_paths = sorted(epoch2path, key=lambda x: x[0])
    sorted_paths = [path for _, path in sorted_paths]

    if reverse:
        sorted_paths = sorted_paths[::-1]

    for name in required_names:
        if name in name2path:
            sorted_paths.append(name2path[name])

    return sorted_paths

def get_ckp_list(exp_dir, summary_path, reverse=False):
    #all_ckp_path_list = glob(os.path.join(exp_dir, "*.ckpt"))
    all_ckp_path_list = glob(os.path.join(exp_dir, "model_final.ckpt"))
    
    all_ckp_path_list = sort_ckp_paths(all_ckp_path_list, reverse=reverse)

    # If there is no summary file, we need to evaluate all the checkpoints

    return all_ckp_path_list
    # if not os.path.exists(summary_path):
    #     return all_ckp_path_list

    all_epochs = [get_ckp_name(ckp_path) for ckp_path in all_ckp_path_list]

    df = pd.read_csv(summary_path)
    evaluated_epochs = set([str(e) for e in df['epoch'].tolist()])  # set(str)

    ckp_to_eval = []
    for epoch, path in zip(all_epochs, all_ckp_path_list):
        if epoch not in evaluated_epochs:
            ckp_to_eval.append(path)

    return ckp_to_eval


def save_success_rate(epoch, success_metrics, summary_file_path):
    success_metrics = {k.replace("rollout/", ""): v for k, v in success_metrics.items()}
    success_heads = list(success_metrics.keys())
    success_heads.remove("success_env_avg")
    success_heads = sorted(success_heads, key=lambda x: int(x.split("success_env")[-1]))
    success_heads.append("success_env_avg")
    success_heads = ["epoch"] + success_heads

    success_metrics["epoch"] = str(epoch)
    df = pd.DataFrame(success_metrics, index=[0])

    if os.path.exists(summary_file_path):
        old_summary = pd.read_csv(summary_file_path)
        df = pd.concat([df, old_summary], ignore_index=True)

    df = df[success_heads]
    df.to_csv(summary_file_path)


def evaluate(fabric, cfg, checkpoint, video_save_dir, num_env_rollouts=20, render_image_size=None, video_speedup=1,
             save_all_video=False, success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
    os.makedirs(video_save_dir, exist_ok=True)
    cfg.model_cfg.load_path = checkpoint

    # breakpoint()

    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)

    cfg.optimizer_cfg.params.lr = 0.
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)

    model, optimizer = fabric.setup(model, optimizer)

    env_type = cfg.env_cfg.env_type
    rollout_horizon = cfg.env_cfg.get("horizon", None)
    # initialize the environments in each rank
    cfg.env_cfg.render_gpu_ids = cfg.env_cfg.render_gpu_ids[fabric.global_rank] if isinstance(cfg.env_cfg.render_gpu_ids, list) else cfg.env_cfg.render_gpu_ids
    env_num_each_rank = math.ceil(len(cfg.env_cfg.env_name) / fabric.world_size)
    env_idx_start = env_num_each_rank * fabric.global_rank
    env_idx_end = min(env_num_each_rank * (fabric.global_rank + 1), len(cfg.env_cfg.env_name))

    all_results = []
    for env_idx in range(env_idx_start, env_idx_end):
        print(f"evaluating ckp {checkpoint} on env {env_idx} in ({env_idx_start}, {env_idx_end})")
        env = build_env(img_size=(render_image_size or cfg.img_size), env_idx_start_end=(env_idx, env_idx+1), **cfg.env_cfg)
        result = rollout(env, model, num_env_rollouts=num_env_rollouts // cfg.env_cfg.vec_env_num, horizon=rollout_horizon,
                         return_wandb_video=False,
                         success_vid_first=success_vid_first, fail_vid_first=fail_vid_first,
                         connect_points_with_line=connect_points_with_line)

        # save videos
        video = None
        for k in list(result.keys()):
            if k.startswith("rollout/vis_env"):
                video = result.pop(k)
        assert video is not None
        if save_all_video:
            video = rearrange(video, "B t c h w -> (B t) h w c")
        else:
            video = rearrange(video[0], "t c h w -> t h w c")
        make_grid_video_from_numpy([video], ncol=1, speedup=video_speedup,
                                   output_name=os.path.join(video_save_dir, f"env_{env_idx}.mp4"))

        all_results.append(result)
        del env
    all_results = merge_results(all_results, compute_avg=False)

    del model
    del optimizer
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


@hydra.main(version_base="1.3")
def main(cfg: DictConfig):
    save_path = cfg.save_path
    result_suffix = cfg.get("result_path_suffix", "")
    result_suffix = f"_{result_suffix}" if result_suffix else result_suffix

    eval_result_dir = os.path.join(save_path, f"eval_results{result_suffix}")
    os.makedirs(eval_result_dir, exist_ok=True)

    render_image_size = cfg.get("render_image_size", cfg.img_size)
    num_env_rollouts = cfg.get("num_env_rollouts", 20)
    save_all_video = cfg.get("save_all_video", False)
    success_vid_first = cfg.get("success_vid_first", False)
    fail_vid_first = cfg.get("fail_vid_first", False)
    connect_points_with_line = cfg.get("connect_points_with_line", False)
    video_speedup = cfg.get("video_speedup", 1)

    # currently hardcode
    suite_name = cfg.env_cfg.env_name[0]

    summary_file_path = os.path.join(eval_result_dir, f"summary_{suite_name}.csv")
    ckp_paths_to_eval = get_ckp_list(save_path, summary_file_path, reverse=True)

    setup(cfg)

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), strategy="ddp")
    fabric.launch()

    for ckp_path in ckp_paths_to_eval:
        epoch_name = get_ckp_name(ckp_path)

        gathered_results = [{} for _ in range(fabric.world_size)]
        results = evaluate(fabric, cfg, checkpoint=ckp_path,
                           video_save_dir=os.path.join(eval_result_dir, f"video_{suite_name}/epoch_{epoch_name}"),
                           render_image_size=render_image_size, video_speedup=video_speedup,
                           num_env_rollouts=num_env_rollouts, save_all_video=save_all_video,
                           success_vid_first=success_vid_first, fail_vid_first=fail_vid_first,
                           connect_points_with_line=connect_points_with_line)
        fabric.barrier()
        dist.all_gather_object(gathered_results, results)

        if fabric.is_global_zero:
            gathered_results = merge_results(gathered_results)

            success_metrics = {k: v for k, v in gathered_results.items() if k.startswith("rollout/success_env")}
            save_success_rate(epoch_name, success_metrics, summary_file_path)


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()

























# import hydra
# import math
# from glob import glob
# import pandas as pd
# import torch

# import datetime
# import torch
# torch.distributed.constants._DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=5000)

# import torch.distributed as dist
# import lightning
# from lightning.fabric import Fabric
# import gc

# import os

# from einops import rearrange
# from omegaconf import DictConfig

# from atm.policy import *
# from atm.utils.train_utils import setup_optimizer
# from atm.utils.env_utils import build_env
# from engine.utils import rollout, merge_results
# from atm.utils.video_utils import make_grid_video_from_numpy

# from engine.utils import rollout


# def get_ckp_name(file_path):
#     return os.path.basename(file_path).split('.ckpt')[0].split('_')[-1]


# def sort_ckp_paths(file_list, reverse=False):
#     required_names = ["final", "best"]

#     epoch2path = []
#     name2path = {}
#     for path in file_list:
#         name = get_ckp_name(path)
#         if name.isdigit():
#             # epoch number checkpoint
#             epoch = int(name)
#             epoch2path.append((epoch, path))
#         else:
#             # final / best checkpoint
#             name2path[name] = path

#     sorted_paths = sorted(epoch2path, key=lambda x: x[0])
#     sorted_paths = [path for _, path in sorted_paths]

#     if reverse:
#         sorted_paths = sorted_paths[::-1]

#     for name in required_names:
#         if name in name2path:
#             sorted_paths.append(name2path[name])

#     return sorted_paths


# def get_ckp_list(exp_dir, summary_path, reverse=False):
#     all_ckp_path_list = glob(os.path.join(exp_dir, "model_final.ckpt"))

#     #all_ckp_path_list.append(os.path.join(exp_dir, "model_110.ckpt"))
#     #all_ckp_path_list.append(os.path.join(exp_dir, "model_90.ckpt"))
#     #all_ckp_path_list.append(os.path.join(exp_dir, "model_80.ckpt"))
#     #all_ckp_path_list.append(os.path.join(exp_dir, "model_120.ckpt"))

#     all_ckp_path_list = sort_ckp_paths(all_ckp_path_list, reverse=reverse)

#     # If there is no summary file, we need to evaluate all the checkpoints
#     if not os.path.exists(summary_path):
#         return all_ckp_path_list

#     all_epochs = [get_ckp_name(ckp_path) for ckp_path in all_ckp_path_list]

#     df = pd.read_csv(summary_path)
#     evaluated_epochs = set([str(e) for e in df['epoch'].tolist()])  # set(str)

#     ckp_to_eval = []
#     for epoch, path in zip(all_epochs, all_ckp_path_list):
#         if epoch not in evaluated_epochs:
#             ckp_to_eval.append(path)

#     return ckp_to_eval


# def save_success_rate(epoch, success_metrics, summary_file_path):
#     success_metrics = {k.replace("rollout/", ""): v for k, v in success_metrics.items()}
#     success_heads = list(success_metrics.keys())
#     success_heads.remove("success_env_avg")
#     success_heads = sorted(success_heads, key=lambda x: int(x.split("success_env")[-1]))
#     success_heads.append("success_env_avg")
#     success_heads = ["epoch"] + success_heads

#     success_metrics["epoch"] = str(epoch)
#     df = pd.DataFrame(success_metrics, index=[0])

#     if os.path.exists(summary_file_path):
#         old_summary = pd.read_csv(summary_file_path)
#         df = pd.concat([df, old_summary], ignore_index=True)

#     df = df[success_heads]
#     df.to_csv(summary_file_path)


# def evaluate(fabric, cfg, checkpoint, video_save_dir, num_env_rollouts=20, render_image_size=None, video_speedup=1,
#              save_all_video=False, success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
#     os.makedirs(video_save_dir, exist_ok=True)
#     cfg.model_cfg.load_path = checkpoint



#     #aaa = BCViLTPolicy()
#     bbb = {'load_path': '/mnt/petrelfs/yangjiange/ATM/results/policy/0923_atm-policy_fold-cloth_demo10_2110_seed0/model_final.ckpt', 'obs_cfg': {'obs_shapes': {'rgb': [3, 128, 128], 'tracks': [16, 32, 2]}, 'img_mean': [0.0, 0.0, 0.0], 'img_std': [1.0, 1.0, 1.0], 'num_views': 2, 'extra_states': ["joint_states", "gripper_states"], 'max_seq_len': 10}, 'img_encoder_cfg': {'network_name': 'PatchEncoder', 'patch_size': [8, 8], 'embed_size': 128, 'no_patch_embed_bias': False}, 'language_encoder_cfg': {'network_name': 'MLPEncoder', 'input_size': 768, 'hidden_size': 128, 'num_layers': 1}, 'extra_state_encoder_cfg': {'extra_num_layers': 0, 'extra_hidden_size': 128}, 'track_cfg': {'track_fn': '/mnt/petrelfs/yangjiange/ATM/results/track_transformer/0923_libero_track_transformer_libero-goal_ep301_2015', 'policy_track_patch_size': 16, 'use_zero_track': False}, 'spatial_transformer_cfg': {'num_layers': 7, 'num_heads': 8, 'head_output_size': 120, 'mlp_hidden_size': 256, 'dropout': 0.1, 'spatial_downsample': True, 'spatial_downsample_embed_size': 64, 'use_language_token': False}, 'temporal_transformer_cfg': {'num_layers': 4, 'num_heads': 6, 'head_output_size': 64, 'mlp_hidden_size': 256, 'dropout': 0.1, 'use_language_token': False}, 'policy_head_cfg': {'network_name': 'DeterministicHead', 'output_size': [12], 'hidden_size': 1024, 'num_layers': 2, 'loss_coef': 1.0, 'action_squash': False}}
#     model = BCViLTPolicy(**bbb)
#     print(model)
    

#     print(cfg.model_name)
#     print(cfg.model_cfg)
#     exit()


#     model_cls = eval(cfg.model_name)



#     model = model_cls(**cfg.model_cfg)

#     cfg.optimizer_cfg.params.lr = 0.
#     optimizer = setup_optimizer(cfg.optimizer_cfg, model)

#     model, optimizer = fabric.setup(model, optimizer)

#     env_type = cfg.env_cfg.env_type
#     rollout_horizon = cfg.env_cfg.get("horizon", None)
#     # initialize the environments in each rank
#     cfg.env_cfg.render_gpu_ids = cfg.env_cfg.render_gpu_ids[fabric.global_rank] if isinstance(cfg.env_cfg.render_gpu_ids, list) else cfg.env_cfg.render_gpu_ids
#     env_num_each_rank = math.ceil(len(cfg.env_cfg.env_name) / fabric.world_size)
#     env_idx_start = env_num_each_rank * fabric.global_rank
#     env_idx_end = min(env_num_each_rank * (fabric.global_rank + 1), len(cfg.env_cfg.env_name))

#     all_results = []
#     for env_idx in range(env_idx_start, env_idx_end):
#         print(f"evaluating ckp {checkpoint} on env {env_idx} in ({env_idx_start}, {env_idx_end})")
#         env = build_env(img_size=(render_image_size or cfg.img_size), env_idx_start_end=(env_idx, env_idx+1), **cfg.env_cfg)
#         result = rollout(env, model, num_env_rollouts=num_env_rollouts // cfg.env_cfg.vec_env_num, horizon=rollout_horizon,
#                          return_wandb_video=False,
#                          success_vid_first=success_vid_first, fail_vid_first=fail_vid_first,
#                          connect_points_with_line=connect_points_with_line)

#         # save videos
#         video = None
#         for k in list(result.keys()):
#             if k.startswith("rollout/vis_env"):
#                 video = result.pop(k)
#         assert video is not None
#         if save_all_video:
#             video = rearrange(video, "B t c h w -> (B t) h w c")
#         else:
#             video = rearrange(video[0], "t c h w -> t h w c")
#         make_grid_video_from_numpy([video], ncol=1, speedup=video_speedup,
#                                    output_name=os.path.join(video_save_dir, f"env_{env_idx}.mp4"))

#         all_results.append(result)
#         del env
#     all_results = merge_results(all_results, compute_avg=False)

#     del model
#     del optimizer
#     torch._C._cuda_clearCublasWorkspaces()
#     gc.collect()
#     torch.cuda.empty_cache()

#     return all_results

# @hydra.main(version_base="1.3")
# def main(cfg: DictConfig):
#     save_path = cfg.save_path
#     result_suffix = cfg.get("result_path_suffix", "")
#     result_suffix = f"_{result_suffix}" if result_suffix else result_suffix

#     eval_result_dir = os.path.join(save_path, f"eval_results{result_suffix}")
#     os.makedirs(eval_result_dir, exist_ok=True)

#     render_image_size = cfg.get("render_image_size", cfg.img_size)
#     num_env_rollouts = cfg.get("num_env_rollouts", 20)
#     save_all_video = cfg.get("save_all_video", False)
#     success_vid_first = cfg.get("success_vid_first", False)
#     fail_vid_first = cfg.get("fail_vid_first", False)
#     connect_points_with_line = cfg.get("connect_points_with_line", False)
#     video_speedup = cfg.get("video_speedup", 1)

#     # currently hardcode
#     suite_name = cfg.env_cfg.env_name[0]

#     summary_file_path = os.path.join(eval_result_dir, f"summary_{suite_name}.csv")
#     ckp_paths_to_eval = get_ckp_list(save_path, summary_file_path, reverse=True)

#     setup(cfg)

#     fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), strategy="ddp")
#     fabric.launch()

#     for ckp_path in ckp_paths_to_eval:
#         epoch_name = get_ckp_name(ckp_path)

#         gathered_results = [{} for _ in range(fabric.world_size)]
#         results = evaluate(fabric, cfg, checkpoint=ckp_path,
#                            video_save_dir=os.path.join(eval_result_dir, f"video_{suite_name}/epoch_{epoch_name}"),
#                            render_image_size=render_image_size, video_speedup=video_speedup,
#                            num_env_rollouts=num_env_rollouts, save_all_video=save_all_video,
#                            success_vid_first=success_vid_first, fail_vid_first=fail_vid_first,
#                            connect_points_with_line=connect_points_with_line)
#         fabric.barrier()
#         dist.all_gather_object(gathered_results, results)

#         if fabric.is_global_zero:
#             gathered_results = merge_results(gathered_results)

#             success_metrics = {k: v for k, v in gathered_results.items() if k.startswith("rollout/success_env")}
#             save_success_rate(epoch_name, success_metrics, summary_file_path)


# def setup(cfg):
#     import warnings

#     warnings.simplefilter("ignore")

#     print("-----------------")
#     print(cfg.seed)

#     lightning.seed_everything(cfg.seed)


# if __name__ == "__main__":
#     main()
