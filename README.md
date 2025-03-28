## [CVPR 2025] [Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](https://arxiv.org/abs/2411.14519)

![caps](./tra_moe.png)

## Low-Cost Dual-Arm Robot Demos

The videos are all done automatically by learned policies.

| Fold | Pick and Pass | Pour |
| --------- | ---------- | ----------- |
| <img src="./real-world demos/folding towels.gif" alt="Fold" height="200"> | <img src="./real-world demos/picking up and passing a holder.gif" alt="Pass" height="200"> | <img src="./real-world demos/pouring water.gif" alt="Pour" height="200"> |

| Pull out | Push |
| --------- | ---------- |
| <img src="./real-world demos/pulling out tissues.gif" alt="Pull out" height="200"> | <img src="./real-world demos/pushing a vegetable to the side of the cutting board.gif" alt="Push" height="200"> |


## Prepare

- Following [ATM](https://github.com/Large-Trajectory-Model/ATM/tree/main), install environment (including [robomimic](https://github.com/ARISE-Initiative/robomimic/tree/5dee58f9cc1235010d0877142b54d0e82dd23986) and [robosuite](https://github.com/ARISE-Initiative/robosuite/tree/eafb81f54ffc104f905ee48a16bb15f059176ad3)).

```
conda env create -f environment.yml
conda activate atm
mkdir third_party & cd third_party
git clone https://github.com/ARISE-Initiative/robomimic.git
git clone https://github.com/ARISE-Initiative/robosuite.git
pip install -e third_party/robosuite/
pip install -e third_party/robomimic/
```



- Downloading and processing [libero](https://github.com/Lifelong-Robot-Learning/LIBERO) data as well as using [Cotracker](https://github.com/facebookresearch/co-tracker) to get trajectory labels.

```
mkdir data
python -m scripts.download_libero_datasets
python -m scripts.preprocess_libero --suite libero_spatial
python -m scripts.preprocess_libero --suite libero_object
python -m scripts.preprocess_libero --suite libero_goal
python -m scripts.preprocess_libero --suite libero_10
python -m scripts.preprocess_libero --suite libero_90
python -m scripts.split_libero_dataset
```

## Training

- Stage 1: Training trajectory prediction models with actionless large-scale out-of-domain video data and small-scale in-domain video data.

```
USE_BFLOAT16=true python -m scripts.train_libero_track_transformer --suite $SUITE_NAME
```

- Stage 2: Training trajectory-guided policy with small-scale in-domain robot data.

```
USE_BFLOAT16=false python -m scripts.train_libero_policy_atm --suite $SUITE_NAME --tt $PATH_TO_TT
```

## Evaluation

```
USE_BFLOAT16=false python -m scripts.eval_libero_policy --suite $SUITE_NAME --exp-dir $PATH_TO_EXP
```

## Citation
Please cite the following paper if you feel this repository useful for your research.
```

```
## Acknowledges

Thanks to the open source of the following projects:

[ATM](https://github.com/Large-Trajectory-Model/ATM/tree/main)

[CoTracker](https://github.com/facebookresearch/co-tracker) 

[mixture-of-experts](https://github.com/lucidrains/mixture-of-experts)

[st-moe-pytorch](https://github.com/lucidrains/st-moe-pytorch/tree/main)

[RealRobot](https://github.com/HaoyiZhu/RealRobot)


