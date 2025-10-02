rl_training

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/KRAnmSnFW3)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/license/bsd-3-clause)

## Tutorial Videos
We've released the following tutorials for training and deploying a reinforcement learning policy. Please check it out! 
- [【四足运控，从入门到精通】第一期 | 从头训练测试部署一个强化学习策略](https://www.bilibili.com/video/BV1xKabz9E2d/?share_source=copy_web&vd_source=57f46145c37bfb96f7583c9e02081590) (Bilibili)  
- [Quadruped Robot Control | From Beginner to Pro Episode 1](https://youtu.be/adrdWCOHzC8?si=sAfsiPRhBu3kEJWo) (YouTube)

## Overview

**rl_training** is a RL training library for deeprobotics robots, based on IsaacLab. The table below lists all available environments:

| Robot Model         | Environment Name (ID)                                      | Screenshot |
|---------------------|------------------------------------------------------------|------------|
| [Deeprobotics Lite3](https://www.deeprobotics.cn/robot/index/product1.html) | Rough-Deeprobotics-Lite3-v0 | <img src="./docs/imgs/deeprobotics_lite3.png" alt="Lite3" width="300">
| [Deeprobotics M20](https://www.deeprobotics.cn/robot/index/lynx.html) | Rough-Deeprobotics-M20-v0 | <img src="./docs/imgs/deeprobotics_m20.png" alt="deeprobotics_m20" width="300">

> [!NOTE]
> If you want to deploy policies in mujoco or real robots, please use the corresponding deploy repo in [Deep Robotics Github Center](https://github.com/DeepRoboticsLab).

## Fork: Lite3 Stairs Project

This repository is a fork of [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training) maintained at [BillCipherrr/rl_training_stairs](https://github.com/BillCipherrr/rl_training_stairs). The fork focuses on extending Lite3 locomotion to stair-climbing terrains while keeping upstream capabilities intact.

- **Project purpose**
  - Provide a reproducible stair-climbing training setup for Deeprobotics Lite3 using Isaac Lab and RSL-RL PPO.
  - Document the workflow so others can train, evaluate, and iterate on the stair task.

- **Planned roadmap**
  1. Keep upstream rough-terrain tasks synced while isolating stair-specific changes.
  2. Finalize the stair terrain generator and environment configuration (`stairs_env_cfg.py`).
  3. Train and benchmark a baseline PPO agent, then iterate on improved hyperparameters / architectures.
  4. Build an evaluation pipeline (headless rollouts, metrics, video capture) for ongoing comparisons.
  5. Publish findings & release tagged builds in the forked repo.

- **Current progress** *(updated 2025-10-01)*
  - Added `DeeproboticsLite3StairsEnvCfg` inheriting from the rough configuration with stair-only terrain generation.
  - Introduced `docs/dev_logs/2025-10-01-stairs-env-sync.md` summarizing implementation steps and validation.
  - Verified configs via `python -m compileall .../stairs_env_cfg.py`; stair task is discoverable through `scripts/tools/list_envs.py`.
  - README, development handbook, and project structure now reflect fork-specific goals and maintenance guidelines.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

  ```bash
  git clone https://github.com/DeepRoboticsLab/rl_training.git
  ```

- Using a python interpreter that has Isaac Lab installed, install the library

  ```bash
  python -m pip install -e source/rl_training
  ```

- Verify that the extension is correctly installed by running the following command to print all the available environments in the extension:

  ```bash
  python scripts/tools/list_envs.py
  ```

<details>

<summary>Setup as Omniverse Extension (Optional, click to expand)</summary>

We provide an example UI extension that will load upon enabling your extension defined in `source/rl_training/rl_training/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `rl_trainingb/source`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

</details>

## Try examples

Deeprobotics Lite3:

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-Lite3-v0 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-Lite3-v0 --num_envs=10
```

Deeprobotics M20:

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-M20-v0 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-M20-v0 --num_envs=10
```

> [!NOTE]
> If you want to control a **SINGLE ROBOT** with the keyboard during playback, add `--keyboard` at the end of the play script.
>
> ```
> Key bindings:
> ====================== ========================= ========================
> Command                Key (+ve axis)            Key (-ve axis)
> ====================== ========================= ========================
> Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
> Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
> Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
> ====================== ========================= ========================
> ```
* Record video of a trained agent (requires installing `ffmpeg`), add `--video --video_length 200`
* Play/Train with 32 environments, add `--num_envs 32`
* Play on specific folder or checkpoint, add `--load_run run_folder_name --checkpoint model.pt`
* Resume training from folder or checkpoint, add `--resume --load_run run_folder_name --checkpoint model.pt`
## Multi-gpu acceleration
* To train with multiple GPUs, use the following command, where --nproc_per_node represents the number of available GPUs:
    ```bash
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless 
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-Lite3-v0 --headless --distributed --num_envs=2048
    ```
* Note: each gpu will have the same number of envs specified in the config, to use the previous total number of envs, devide it by the number of gpus.
* To scale up training beyond multiple GPUs on a single machine, it is also possible to train across multiple nodes. To train across multiple nodes/machines, it is required to launch an individual process on each node.

    For the master node, use the following command, where --nproc_per_node represents the number of available GPUs, and --nnodes represents the number of nodes:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless --distributed
    ```
    Note that the port (`5555`) can be replaced with any other available port.
    For non-master nodes, use the following command, replacing `--node_rank` with the index of each machine:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless --distributed
    ```

## Tensorboard

To view tensorboard, run:

```bash
tensorboard --logdir=logs
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

**Note: Replace `<path-to-isaac-lab>` with your own IsaacLab path.**

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/rl_training",
        "/<path-to-isaac-lab>/source/isaaclab",
        "/<path-to-isaac-lab>/source/isaaclab_assets",
        "/<path-to-isaac-lab>/source/isaaclab_mimic",
        "/<path-to-isaac-lab>/source/isaaclab_rl",
        "/<path-to-isaac-lab>/source/isaaclab_tasks",
    ]
}
```

### Clean USD Caches

Temporary USD files are generated in `/tmp/IsaacLab/usd_{date}_{time}_{random}` during simulation runs. These files can consume significant disk space and can be cleaned by:

```bash
rm -rf /tmp/IsaacLab/usd_*
```

## Acknowledgements

The project uses some code from the following open-source code repositories:

- [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)
