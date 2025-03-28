import os
import argparse
from glob import glob


# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# input parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--suite",
    default="libero_goal",
    choices=["libero_spatial", "libero_object", "libero_goal", "libero_100"],
    help="The name of the desired suite, where libero_10 is the alias of libero_long.",
)
args = parser.parse_args()

# training configs
CONFIG_NAME = "libero_track_transformer"

gpu_ids = [0, 1, 2, 3]
#gpu_ids = [0]

root_dir = "/mnt/hwfile/3dv/haoyi/atm_libero"
suite_name = args.suite


# we train 300 epoch using 2200 demo
EPOCH = 301 if suite_name == "libero_100" else 301

# setup number of epoches and dataset path
if suite_name == "libero_100":
    for suite_n in ["libero_90", "libero_10"]:
        for dirname in os.listdir(os.path.join(root_dir, f"{suite_n}")):
            os.makedirs(
                os.path.join(root_dir, f"{suite_n}", dirname, "train"), exist_ok=True
            )
            os.makedirs(
                os.path.join(root_dir, f"{suite_n}", dirname, "val"), exist_ok=True
            )

    train_dataset_list = glob(os.path.join(root_dir, "libero_90/*/train/")) + glob(
        os.path.join(root_dir, "libero_10/*/train/")
    )
    val1_dataset_list = glob(os.path.join(root_dir, "libero_90/*/val/")) + glob(
        os.path.join(root_dir, "libero_10/*/val/")
    )
else:


    aaa = glob("/mnt/hwfile/internvideo/share_data/shiyansong/atm_rlbench/*/")



    train_dataset_list = glob(os.path.join(root_dir, "libero_goal/*/train/")) + glob(os.path.join(root_dir, "libero_spatial/*/train/")) + glob(os.path.join(root_dir, "libero_object/*/train/")) + glob(os.path.join(root_dir, "libero_10/*/train/")) + glob(os.path.join(root_dir, "libero_90/*/"))

    val1_dataset_list =  glob(os.path.join(root_dir, "libero_goal/*/val/")) + glob(os.path.join(root_dir, "libero_spatial/*/val/")) + glob(os.path.join(root_dir, "libero_object/*/val/")) + glob(os.path.join(root_dir, "libero_10/*/val/"))


    ##+ [path for path in aaa if ('weighing_scales' not in path and 'open_jar' not in path and 'put_all_groceries_in_cupboard' not in path and "empty_dishwasher" not in path)]
    #glob(os.path.join(root_dir, "libero_10/*/train/")) + glob(os.path.join(root_dir, "libero_90/*/")) + [path for path in aaa if ('weighing_scales' not in path and 'open_jar' not in path and 'put_all_groceries_in_cupboard' not in path and "empty_dishwasher" not in path)]



command = (
    f"python -m engine.train_track_transformer --config-name={CONFIG_NAME} "
    f'train_gpus="{gpu_ids}" '
    f'experiment={CONFIG_NAME}_{suite_name.replace("_", "-")}_ep{EPOCH} '
    f"epochs={EPOCH} "
    f'train_dataset="{train_dataset_list}" val_dataset="{val1_dataset_list}" '
)

os.system(command)
