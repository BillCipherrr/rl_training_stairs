# 歷史觀測輸入設計文件

> **文件版本**：v1.0  
> **建立日期**：2026-02-04  
> **狀態**：設計完成，待實作

---

## 1. 概述

### 1.1 目標

將 PPO 策略網路的輸入從單一時間步觀測（45維）擴展為包含歷史資訊的時序觀測（900維），以捕捉約兩個完整步態週期的動態資訊。

### 1.2 動機

- **時序資訊**：單幀觀測無法區分「正在加速」與「正在減速」等動態狀態
- **步態週期**：四足機器人的步態週期約 200ms（10步 @ 50Hz），20步歷史可涵蓋兩個完整週期
- **Sim-to-Real**：歷史資訊有助於策略在真實世界中應對感測器延遲與噪聲

---

## 2. 設計規格

### 2.1 核心參數

| 參數 | 值 | 說明 |
|------|-----|------|
| **history_length** | 20 | 保留最近 20 個時間步（含當前） |
| **單步觀測維度** | 45 | 現有 Policy 輸入維度 |
| **總輸入維度** | 900 | 45 × 20 = 900 |
| **時間窗口** | 400ms | 20 步 × 20ms/步 = 400ms |
| **步態覆蓋** | ~2 週期 | Lite3 步態週期約 200ms |

### 2.2 觀測組成（單步 45 維）

| 觀測項目 | 維度 | 說明 |
|----------|------|------|
| `base_ang_vel` | 3 | 基座角速度 |
| `projected_gravity` | 3 | 投影重力向量 |
| `velocity_commands` | 3 | 速度指令 (vx, vy, ω) |
| `joint_pos` | 12 | 關節位置（相對默認） |
| `joint_vel` | 12 | 關節速度 |
| `actions` | 12 | 上一時刻動作 |
| **合計** | **45** | |

> **注意**：`base_lin_vel` 和 `height_scan` 在 Lite3 Rough 配置中被禁用

### 2.3 歷史展開結構

當 `flatten_history_dim=True` 時，輸出為 2D 張量：

```
輸出形狀: (num_envs, 900)

記憶體佈局 (LIFO - 最新在最後):
[t-19的obs(45維)][t-18的obs(45維)]...[t-1的obs(45維)][t的obs(45維)]
     ↑ 最舊                                              ↑ 最新（當前）
```

### 2.4 重置行為

| 事件 | 行為 |
|------|------|
| 環境初始化 | 緩衝區以第一個有效觀測填充（Repeat-First） |
| Episode 重置 | 對應環境的緩衝區清空並重新填充 |
| 部分環境重置 | 僅重置指定 `env_ids` 的緩衝區 |

---

## 3. 實作方案

### 3.1 方案選擇

採用 **Isaac Lab 內建 `history_length`** 機制，在 `ObservationGroupCfg` 層級統一設定。

**優點**：
- 零額外程式碼，利用框架原生支援
- 自動處理環境重置時的緩衝區管理
- 與現有訓練流程完全兼容

### 3.2 檔案修改清單

| 檔案 | 修改類型 | 說明 |
|------|----------|------|
| `velocity_env_cfg.py` | 新增類別 | 新增 `PolicyHistoryCfg` 觀測群組 |
| `rough_env_cfg.py` | 新增子類 | 新增 `DeeproboticsLite3RoughHistoryEnvCfg` |
| `__init__.py` | 新增註冊 | 註冊新的 Gym 環境 ID |
| `agents/rsl_rl_ppo_cfg.py` | 新增配置 | 新增對應的 PPO Runner 配置 |

### 3.3 向後兼容策略

- **不修改現有類別**：所有歷史功能透過繼承實作
- **新增環境 ID**：`Isaac-Velocity-Rough-Deeprobotics-Lite3-History-v0`
- **原環境保留**：`Isaac-Velocity-Rough-Deeprobotics-Lite3-v0` 維持 45 維輸入

---

## 4. 程式碼範例

### 4.1 基礎環境配置 (velocity_env_cfg.py)

```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (current frame only)."""
        # ... 現有定義 ...
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyHistoryCfg(ObsGroup):
        """Observations for policy group with history."""
        
        # 與 PolicyCfg 相同的觀測項目
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 20          # 保留 20 個時間步
            self.flatten_history_dim = True   # 展平為 (N, 900)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
```

### 4.2 機器人專屬配置 (rough_env_cfg.py)

```python
@configclass
class DeeproboticsLite3RoughHistoryEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """Lite3 rough terrain environment with observation history."""
    
    def __post_init__(self):
        # 呼叫父類初始化
        super().__post_init__()
        
        # 切換到歷史觀測群組
        self.observations.policy = ObservationsCfg.PolicyHistoryCfg()
        
        # 重新套用 Lite3 專屬的觀測設定
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
```

### 4.3 環境註冊 (__init__.py)

```python
gym.register(
    id="Isaac-Velocity-Rough-Deeprobotics-Lite3-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsLite3RoughHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughHistoryPPORunnerCfg",
    },
)
```

### 4.4 PPO 配置 (rsl_rl_ppo_cfg.py)

```python
@configclass
class DeeproboticsLite3RoughHistoryPPORunnerCfg(DeeproboticsLite3RoughPPORunnerCfg):
    """PPO configuration for Lite3 with observation history."""
    
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_lite3_rough_history"
        
        # 可選：根據後續 TensorBoard 分析調整
        # self.policy.actor_hidden_dims = [1024, 512, 256, 128]
        # self.policy.critic_hidden_dims = [1024, 512, 256, 128]
```

---

## 5. 驗證步驟

### 5.1 維度驗證

```bash
# 啟動環境並檢查觀測維度
python -c "
import gymnasium as gym
import rl_training.tasks  # 註冊環境

env = gym.make('Isaac-Velocity-Rough-Deeprobotics-Lite3-History-v0', num_envs=1)
obs, _ = env.reset()
print(f'Observation shape: {obs[\"policy\"].shape}')  # 預期: (1, 900)
env.close()
"
```

### 5.2 歷史緩衝驗證

```python
# 在訓練腳本中加入驗證
def verify_history_buffer(env):
    """驗證歷史緩衝區行為"""
    obs1, _ = env.reset()
    obs2, _, _, _, _ = env.step(env.action_space.sample())
    obs3, _, _, _, _ = env.step(env.action_space.sample())
    
    # 檢查最後 45 維（最新觀測）是否與 obs3 的前 45 維不同
    # 因為歷史已經移動
    print(f"History buffer updated: {not torch.allclose(obs2['policy'][:, -45:], obs3['policy'][:, -45:])}")
```

### 5.3 訓練啟動

```bash
# 使用新環境進行訓練
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Deeprobotics-Lite3-History-v0 \
    --num_envs 4096 \
    --headless
```

---

## 6. 監控指標

### 6.1 TensorBoard 關注項目

| 指標 | 預期行為 | 異常徵兆 |
|------|----------|----------|
| `Loss/value_function` | 穩定下降 | 震盪或發散 → 考慮增加網路容量 |
| `Loss/surrogate` | 穩定 | 劇烈波動 → 降低 learning rate |
| `Perf/mean_reward` | 緩慢上升 | 長期停滯 → 網路容量不足 |
| `Train/mean_episode_length` | 逐漸增加 | 無改善 → 歷史資訊未被有效利用 |

### 6.2 網路容量調整建議

若觀察到訓練瓶頸，考慮以下調整：

```python
# 漸進式擴容方案
policy = RslRlPpoActorCriticCfg(
    # 方案 A：增加第一層寬度
    actor_hidden_dims=[1024, 512, 256, 128],
    critic_hidden_dims=[1024, 512, 256, 128],
    
    # 方案 B：全面擴容
    # actor_hidden_dims=[1024, 512, 512, 256],
    # critic_hidden_dims=[1024, 512, 512, 256],
)
```

---

## 7. 資源估算

### 7.1 記憶體用量

| 項目 | 原始 (45維) | 歷史 (900維) | 增幅 |
|------|-------------|--------------|------|
| 單環境觀測 | 180 bytes | 3,600 bytes | 20x |
| 4096 環境 | 0.7 MB | 14.1 MB | 20x |
| Rollout Buffer (24步) | 17.6 MB | 352 MB | 20x |

### 7.2 訓練速度影響

- **前向傳播**：輸入維度增加，但 MLP 結構不變，影響有限（預估 +10-20%）
- **GPU 記憶體**：主要增量來自觀測緩衝區，非瓶頸
- **總體影響**：預估訓練速度下降 15-25%

---

## 8. 未來擴展

### 8.1 選擇性歷史（進階優化）

若發現某些觀測項目的歷史價值較低，可改為選擇性歷史：

```python
# 僅對關節相關項目保留歷史
joint_pos = ObsTerm(..., history_length=20)
joint_vel = ObsTerm(..., history_length=10)
actions = ObsTerm(..., history_length=20)
# 其他項目不保留歷史
base_ang_vel = ObsTerm(..., history_length=0)
```

### 8.2 RNN/Transformer 整合

若未來需要更強的時序建模能力：

```python
# 使用 flatten_history_dim=False 保留時序維度
def __post_init__(self):
    self.history_length = 20
    self.flatten_history_dim = False  # 輸出: (N, 20, 45)
```

需搭配自訂 Actor-Critic 網路，包含 LSTM 或 Transformer 層。

---

## 9. 附錄

### A. 相關檔案路徑

```
source/rl_training/rl_training/
├── tasks/manager_based/locomotion/velocity/
│   ├── velocity_env_cfg.py                    # 基礎環境配置
│   ├── mdp/
│   │   └── observations.py                    # 觀測函數
│   └── config/quadruped/deeprobotics_lite3/
│       ├── rough_env_cfg.py                   # Lite3 環境配置
│       ├── __init__.py                        # 環境註冊
│       └── agents/
│           └── rsl_rl_ppo_cfg.py              # PPO 配置
```

### B. Isaac Lab 相關 API

- `ObservationTermCfg.history_length`：單項觀測歷史長度
- `ObservationGroupCfg.history_length`：群組級歷史長度（覆蓋所有項目）
- `ObservationGroupCfg.flatten_history_dim`：是否展平歷史維度
- `CircularBuffer`：底層環形緩衝區實作

### C. 參考資料

- Isaac Lab Observation Manager：`isaaclab/managers/observation_manager.py`
- Circular Buffer：`isaaclab/utils/buffers/circular_buffer.py`
- RSL-RL Actor-Critic：`rsl_rl/modules/actor_critic.py`
