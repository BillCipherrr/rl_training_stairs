# Lite3 樓梯環境開發手冊

## 概述

本文件說明 DeepRobotics Lite3 四足機器人樓梯攀爬環境的設計與使用方式。

### 新增環境

| 環境 ID | 說明 | 用途 |
|---------|------|------|
| `Stairs-Deeprobotics-Lite3-v0` | 基礎樓梯環境 | 從零開始訓練 |
| `Stairs-Deeprobotics-Lite3-History-v0` | 樓梯環境 + 20步歷史觀測 | 從預訓練模型微調 |

---

## 1. 環境設計

### 1.1 地形配置

```
STAIRS_TERRAINS_CFG
├── pyramid_stairs (25%)        - 上樓梯
├── pyramid_stairs_inv (25%)    - 下樓梯
├── hf_pyramid_slope (10%)      - 上坡
├── hf_pyramid_slope_inv (10%)  - 下坡
└── random_rough (30%)          - 崎嶇地形
```

#### 樓梯參數
| 參數 | 數值 | 說明 |
|------|------|------|
| 階高範圍 | 0.05 ~ 0.20 m | 課程學習：由低到高 |
| 階寬 | 0.28 m | 標準階寬 |
| 平台寬度 | 3.0 m | 樓梯頂部平台 |

### 1.2 觀測空間

**Actor Policy 觀測（兩版本相同）**
- 角速度 (3)
- 重力投影 (3)
- 速度指令 (3)
- 關節位置 (12)
- 關節速度 (12)
- 上一步動作 (12)
- **總計：45 維**

> 注意：Height Scan 不包含在 actor policy 中，僅供 critic 使用。

**歷史版 (Stairs-History-v0)**
- 上述觀測 × 20 timesteps = **900 維**
- 提供時序資訊以預測地形變化

### 1.3 獎勵函數調整

相較於 `Rough` 環境，樓梯環境做了以下調整：

| 獎勵項 | Rough 權重 | Stairs 權重 | 調整原因 |
|--------|------------|-------------|----------|
| `feet_air_time` | 5.0 | 5.0 | 維持明確抬腿 |
| `feet_height` | -0.2 | **-0.5** | 鼓勵更高抬腿以跨過階梯 |
| `feet_stumble` | 0.0 | **-1.0** | 懲罰腳撞到階梯邊緣 |
| `flat_orientation_l2` | -5.0 | **-2.0** | 允許爬梯時身體傾斜 |
| `track_lin_vel_xy_exp` | 3.0 | **2.5** | 降低速度優先權，增加穩定性 |
| `feet_slide` | -0.05 | **-0.1** | 防止在階梯邊緣滑動 |
| `base_height_l2` | -10.0 | **-8.0** | 略微放寬高度限制 |

### 1.4 課程學習

#### 地形難度課程 (`terrain_levels`)
- 自動根據機器人表現調整地形難度
- 低難度：低階梯、簡單地形
- 高難度：高階梯、複雜地形

#### 速度指令課程 (`command_levels`)
- 初始：最大速度的 30%
- 最終：最大速度的 100%
- 依據 `track_lin_vel_xy_exp` 獎勵表現調整

### 1.5 速度指令範圍

| 參數 | Rough 環境 | Stairs 環境 | 說明 |
|------|------------|-------------|------|
| `lin_vel_x` | ±1.5 m/s | **±1.0 m/s** | 較保守的前進速度 |
| `lin_vel_y` | ±0.8 m/s | **±0.5 m/s** | 較保守的側移速度 |
| `ang_vel_z` | ±0.8 rad/s | **±0.6 rad/s** | 較保守的轉彎速度 |

---

## 2. 訓練參數

### 2.1 PPO 配置

**基礎版 (StairsPPORunnerCfg)**
```python
num_steps_per_env = 32        # 每環境步數（增加以處理複雜地形）
max_iterations = 20000        # 最大迭代數（課程學習需要更長時間）
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
learning_rate = 1.0e-3
entropy_coef = 0.01
```

**歷史版 (StairsHistoryPPORunnerCfg)**
```python
num_steps_per_env = 32
max_iterations = 20000
actor_hidden_dims = [1024, 512, 256, 128]   # 更大網路處理歷史輸入
critic_hidden_dims = [1024, 512, 256, 128]
learning_rate = 1.0e-3        # 微調時建議改為 1.0e-4
entropy_coef = 0.008          # 微調時略低以保持穩定
```

---

## 3. 訓練指令

### 3.1 從零開始訓練（基礎版）

```bash
cd /home/user1/rl_training

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Stairs-Deeprobotics-Lite3-v0 \
    --headless \
    --num_envs 4096
```

### 3.2 從預訓練模型微調（推薦）

使用 `Rough-History` 預訓練模型作為起點：

```bash
cd /home/user1/rl_training

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Stairs-Deeprobotics-Lite3-History-v0 \
    --headless \
    --num_envs 4096 \
    --load_run 2026-02-04_19-09-09 \
    --checkpoint model_9999.pt
```

**微調建議**：
- 可考慮降低學習率至 `1e-4`（需修改 PPO 配置或透過命令列參數）
- 初期可減少 `num_envs` 以便更快迭代觀察

### 3.3 測試/播放訓練結果

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Stairs-Deeprobotics-Lite3-History-v0 \
    --num_envs 64 \
    --load_run <timestamp> \
    --checkpoint model_<iter>.pt
```

---

## 4. 檔案結構

```
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/
├── config/quadruped/deeprobotics_lite3/
│   ├── __init__.py              # 環境註冊（已更新）
│   ├── flat_env_cfg.py          # 平面環境
│   ├── rough_env_cfg.py         # 崎嶇環境
│   ├── stairs_env_cfg.py        # 【新增】樓梯環境
│   └── agents/
│       └── rsl_rl_ppo_cfg.py    # PPO 配置（已更新）
└── velocity_env_cfg.py          # 基礎環境配置
```

---

## 5. 進階調整建議

### 5.1 如果機器人跌倒率過高

1. **降低速度指令範圍**
   ```python
   self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
   ```

2. **增加穩定性相關獎勵權重**
   ```python
   self.rewards.flat_orientation_l2.weight = -3.0  # 增加
   self.rewards.feet_slide.weight = -0.2           # 增加
   ```

3. **降低課程學習起始難度**
   ```python
   self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
   ```

### 5.2 如果爬樓梯速度太慢

1. **提高速度追蹤獎勵**
   ```python
   self.rewards.track_lin_vel_xy_exp.weight = 3.0
   ```

2. **調整 feet_air_time 閾值**
   ```python
   self.rewards.feet_air_time.params["threshold"] = 0.35  # 降低
   ```

### 5.3 如果腳經常撞到階梯

1. **增加 feet_stumble 懲罰**
   ```python
   self.rewards.feet_stumble.weight = -2.0
   ```

2. **增加抬腳高度獎勵**
   ```python
   self.rewards.feet_height.weight = -1.0
   self.rewards.feet_height.params["target_height"] = 0.10
   ```

---

## 6. 預期訓練時程

| 階段 | 迭代數 | 預期行為 |
|------|--------|----------|
| 0 - 2000 | 初期探索 | 學習基本站立和移動 |
| 2000 - 5000 | 適應地形 | 開始適應簡單樓梯 |
| 5000 - 10000 | 課程提升 | 逐步處理更高階梯 |
| 10000 - 15000 | 穩定優化 | 提升爬樓梯穩定性 |
| 15000 - 20000 | 精細調整 | 最佳化步態和效率 |

---

## 7. 常見問題

### Q1: 為什麼選擇 50% 樓梯而非 100%？
混合地形有助於：
- 防止過擬合到特定樓梯模式
- 保持在平地和斜坡上的泛化能力
- 課程學習更平滑（從簡單地形過渡）

### Q2: 為什麼需要 History 版本？
歷史觀測提供時序資訊，讓策略網路能夠：
- 估計當前運動趨勢
- 預測即將到來的地形變化
- 更好地規劃腳步放置

### Q3: 微調 vs 從零訓練，哪個更好？
**推薦微調**，原因：
- Rough 環境已學會基本移動技能
- 歷史觀測處理能力可直接遷移
- 訓練時間可大幅縮短

---

## 8. 相關檔案路徑

| 檔案 | 路徑 |
|------|------|
| 樓梯環境配置 | `source/rl_training/.../deeprobotics_lite3/stairs_env_cfg.py` |
| PPO 配置 | `source/rl_training/.../deeprobotics_lite3/agents/rsl_rl_ppo_cfg.py` |
| 預訓練模型 | `logs/rsl_rl/deeprobotics_lite3_rough_history/2026-02-04_19-09-09/model_9999.pt` |
| 訓練腳本 | `scripts/reinforcement_learning/rsl_rl/train.py` |

---

*文件建立日期：2026-02-05*
*適用版本：rl_training with IsaacLab*
