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

---

## 10. 實作記錄

> **實作日期**：2026-02-04  
> **實作者**：GitHub Copilot (Claude Sonnet 4.5)  
> **狀態**：✅ 實作完成並驗證通過

### 10.1 實作摘要

根據本設計文件完成了 Lite3 機器人的歷史觀測功能實作，所有修改遵循向後兼容原則，未影響現有環境。

### 10.2 檔案修改清單

| 檔案路徑 | 修改內容 | 行數變化 |
|----------|----------|----------|
| `velocity_env_cfg.py` | 新增 `PolicyHistoryCfg` 類別 | +70 |
| `rough_env_cfg.py` | 新增 `DeeproboticsLite3RoughHistoryEnvCfg` 類別 | +41 |
| `__init__.py` | 註冊新環境 `Rough-Deeprobotics-Lite3-History-v0` | +10 |
| `rsl_rl_ppo_cfg.py` | 新增 `DeeproboticsLite3RoughHistoryPPORunnerCfg` 類別 | +22 |

### 10.3 實際實作細節

#### 10.3.1 PolicyHistoryCfg 定義

```python
@configclass
class PolicyHistoryCfg(ObsGroup):
    """Observations for policy group with history (20 timesteps).
    
    This observation group maintains a history buffer of the last 20 timesteps,
    covering approximately two complete gait cycles (~400ms at 50Hz).
    The output is flattened to (num_envs, 900) where 900 = 45 dims × 20 steps.
    """
    
    # 與 PolicyCfg 相同的觀測項定義
    # ...
    
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 20          # Keep 20 timesteps (~400ms at 50Hz)
        self.flatten_history_dim = True   # Flatten to (N, 900)
```

#### 10.3.2 Lite3 History 環境配置

```python
@configclass
class DeeproboticsLite3RoughHistoryEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """Lite3 rough terrain environment with observation history."""
    
    def __post_init__(self):
        from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
        from isaaclab.managers import SceneEntityCfg
        
        super().__post_init__()
        
        # 切換到歷史觀測群組
        self.observations.policy = ObservationsCfg.PolicyHistoryCfg()
        
        # 重新套用 Lite3 專屬設定
        self.observations.policy.base_lin_vel = None  # 禁用
        self.observations.policy.height_scan = None   # 禁用
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        
        # 重新設定關節名稱
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.joint_names, preserve_order=True
        )
        
        # 禁用 weight=0 的 rewards（修復配置錯誤）
        self.disable_zero_weight_rewards()
```

#### 10.3.3 PPO 網路配置

```python
@configclass
class DeeproboticsLite3RoughHistoryPPORunnerCfg(DeeproboticsLite3RoughPPORunnerCfg):
    """PPO configuration for Lite3 with observation history."""
    
    experiment_name = "deeprobotics_lite3_rough_history"
    
    # 增加網路容量以處理 900 維輸入
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[1024, 512, 256, 128],   # 第一層從 512 增至 1024
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )
```

### 10.4 驗證結果

#### 環境註冊驗證

```bash
✅ 環境 ID: Rough-Deeprobotics-Lite3-History-v0
✅ Entry Point: ManagerBasedRLEnv
✅ Config: DeeproboticsLite3RoughHistoryEnvCfg
```

#### 觀測維度驗證

```
Active Observation Terms in Group: 'policy' (shape: (900,))
+-----------+--------------------------------+-------------+
|   Index   | Name                           |    Shape    |
+-----------+--------------------------------+-------------+
|     0     | base_ang_vel                   |    (60,)    |  ✅ 3×20=60
|     1     | projected_gravity              |    (60,)    |  ✅ 3×20=60
|     2     | velocity_commands              |    (60,)    |  ✅ 3×20=60
|     3     | joint_pos                      |    (240,)   |  ✅ 12×20=240
|     4     | joint_vel                      |    (240,)   |  ✅ 12×20=240
|     5     | actions                        |    (240,)   |  ✅ 12×20=240
+-----------+--------------------------------+-------------+
Total: 900 dims ✅
```

**確認項目**：
- ✅ `base_lin_vel` 已正確禁用（未出現在列表中）
- ✅ `height_scan` 已正確禁用（未出現在列表中）
- ✅ 歷史展平成功（每個觀測項 × 20）
- ✅ 總維度正確（900 = 45 × 20）

#### 網路輸入驗證

```
Actor MLP: MLP(
  (0): Linear(in_features=900, out_features=1024, bias=True)  ✅
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=1024, out_features=512, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=512, out_features=256, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=256, out_features=128, bias=True)
  ...
)
```

**確認項目**：
- ✅ Actor 網路輸入維度為 900
- ✅ 網路架構為 [1024, 512, 256, 128]
- ✅ 第一層寬度已從 512 增加至 1024

#### 訓練測試驗證

```bash
# 測試指令
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Deeprobotics-Lite3-History-v0 \
    --num_envs 64 \
    --headless \
    --max_iterations 5

# 執行結果
✅ 環境成功載入（64 個並行環境）
✅ 訓練循環正常執行（5 iterations 完成）
✅ 無配置錯誤或維度不匹配
✅ Reward 統計正常輸出
✅ 訓練速度：~2500 steps/s
```

**訓練輸出片段**：
```
Learning iteration 4/5
Computation: 2532 steps/s (collection: 0.555s, learning 0.052s)
Mean reward: -1.55
Mean episode length: 51.75
Episode_Reward/track_lin_vel_xy_exp: 0.1654
Episode_Reward/track_ang_vel_z_exp: 0.0521
...
```

### 10.5 問題修復記錄

#### 問題 1：body_lin_acc_l2 配置錯誤

**症狀**：
```
ValueError: Error while parsing 'body_lin_acc_l2:asset_cfg'. 
Not all regular expressions are matched! ... : []
```

**原因**：`DeeproboticsLite3RoughEnvCfg` 僅在類別名稱為 `"DeeproboticsLite3RoughEnvCfg"` 時呼叫 `disable_zero_weight_rewards()`，導致繼承類別無法禁用 weight=0 的 rewards。

**解決方案**：在 `DeeproboticsLite3RoughHistoryEnvCfg.__post_init__()` 結尾明確呼叫 `self.disable_zero_weight_rewards()`。

### 10.6 驗證清單

| 檢查項目 | 狀態 | 備註 |
|----------|------|------|
| PolicyHistoryCfg 定義 | ✅ | 包含 history_length=20, flatten_history_dim=True |
| History 環境配置 | ✅ | 正確繼承並切換觀測群組 |
| Lite3 專屬設定 | ✅ | base_lin_vel/height_scan 已禁用，scales 已套用 |
| 環境註冊 | ✅ | Rough-Deeprobotics-Lite3-History-v0 |
| PPO 配置 | ✅ | 網路架構 [1024, 512, 256, 128] |
| 觀測維度 | ✅ | Policy: 900, Critic: 393 |
| 歷史展開 | ✅ | 每個 obs 項目正確 ×20 |
| 訓練執行 | ✅ | Headless 模式正常運行 |
| 向後兼容 | ✅ | 原環境 Rough-Deeprobotics-Lite3-v0 不受影響 |
| 錯誤修復 | ✅ | body_lin_acc_l2 配置問題已解決 |

### 10.7 使用範例

#### 基礎訓練

```bash
# 使用歷史觀測環境訓練
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Deeprobotics-Lite3-History-v0 \
    --num_envs 4096 \
    --headless
```

#### 與原始環境比較

```bash
# 原始環境（45 維）
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Deeprobotics-Lite3-v0 \
    --num_envs 4096 \
    --headless

# 歷史觀測環境（900 維）
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Deeprobotics-Lite3-History-v0 \
    --num_envs 4096 \
    --headless
```

### 10.8 性能影響

| 指標 | 原始環境 (45維) | 歷史環境 (900維) | 差異 |
|------|-----------------|------------------|------|
| 觀測維度 | 45 | 900 | +1900% |
| Policy 網路第一層 | 512 | 1024 | +100% |
| 訓練速度（估計） | ~3000 steps/s | ~2500 steps/s | -17% |
| GPU 記憶體（4096 envs） | ~18 MB | ~352 MB | +1855% |

**備註**：訓練速度下降幅度在可接受範圍內，主要瓶頸在觀測緩衝區而非網路計算。

### 10.9 下一步建議

1. **性能監控**：使用 TensorBoard 比較歷史觀測與原始環境的訓練曲線
2. **網路調優**：根據訓練表現調整 Actor-Critic 網路架構
3. **超參數搜索**：可能需要調整 learning rate、batch size 等
4. **Sim-to-Real**：驗證歷史觀測在真實機器人上的遷移效果
5. **選擇性歷史**：若發現某些觀測項的歷史價值較低，可進一步優化

### 10.10 相關資源

- **日誌目錄**：`/home/user1/rl_training/logs/rsl_rl/deeprobotics_lite3_rough_history`
- **Checkpoint**：`logs/rsl_rl/deeprobotics_lite3_rough_history/<date>/model_*.pt`
- **TensorBoard**：`tensorboard --logdir logs/rsl_rl/deeprobotics_lite3_rough_history`

---

**總結**：歷史觀測功能已成功實作並通過驗證，可以開始進行完整訓練與性能評估。

