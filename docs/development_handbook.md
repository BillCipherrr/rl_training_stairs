# Lite3 強化學習開發手冊

## 目標概述
- 建立可隨機化階梯地形，支援 Lite3 腳式機器人訓練。
- 以現有 PPO 設定作為基線，完成訓練與量化評估。
- 開發改良版 PPO，重新訓練並比較兩套 policy 的表現。
- 建立評測流程，便於之後持續優化與復現。

## 專案架構速覽
| 路徑 | 說明 |
| --- | --- |
| `scripts/reinforcement_learning/rsl_rl/train.py` | 透過 Hydra 載入環境 & agent 設定後，使用 `OnPolicyRunner` 執行訓練 |
| `scripts/reinforcement_learning/rsl_rl/play.py` | 載入 checkpoint 進行推論、錄影、鍵盤操控 |
| `source/rl_training/rl_training/tasks/.../velocity_env_cfg.py` | 定義環境、場景、感測、獎勵、事件等設定 |
| `source/rl_training/rl_training/tasks/.../agents/` | 各任務的訓練演算法（PPO）設定 |
| `logs/`、`outputs/` | 訓練過程與 Hydra log 存放位置 |

## 開發環境準備
1. **Isaac Lab / Isaac Sim**
   - 依官方文件安裝，建議採 conda 環境。
   - 確保 Isaac Lab Python 模組可供匯入（`isaaclab`, `isaaclab_tasks`, `isaaclab_rl`...）。
2. **Clone 與安裝本專案**
   ```bash
   git clone https://github.com/DeepRoboticsLab/rl_training.git
   cd rl_training
   python -m pip install -e source/rl_training
   ```
3. **驗證環境註冊**
   ```bash
   python scripts/tools/list_envs.py
   ```
   確認 `Rough-Deeprobotics-Lite3-v0` 等環境可被列出。

## 階梯地形設計
1. **建立自訂地形配置**
   - 新增 `StairsSceneCfg`：繼承 `InteractiveSceneCfg`，以 `TerrainImporterCfg` 或自定 heightfield 生成階梯。
   - 隨機化參數：階梯高度、深度、摩擦係數，可放在 `terrain_generator` 或 `events` 階段抽樣。
2. **整合到環境**
   - 在 `velocity_env_cfg.py` 內加入新的環境設定類別，例如 `DeeproboticsLite3StairsEnvCfg`，將 `scene` 替換為 `StairsSceneCfg`。
   - 更新 `__init__.py` 或相應 registry，註冊新的 task id（例如 `Stairs-Deeprobotics-Lite3-v0`）。
3. **Domain Randomization 提示**
   - `EventCfg` 中加入階梯高度/摩擦的 reset 隨機化。
   - 若需控制初始位置，調整 `randomize_reset_base` 以對齊階梯起點。

## 基線 PPO 訓練流程
1. **使用既有 PPO Config**
   - `DeeproboticsLite3RoughPPORunnerCfg` 為基線，可拷貝並調整 `experiment_name`、`max_iterations`。
2. **啟動訓練**
   ```bash
   python scripts/reinforcement_learning/rsl_rl/train.py \
     --task=Stairs-Deeprobotics-Lite3-v0 \
     --agent=rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.agents.rsl_rl_ppo_cfg.DeeproboticsLite3StairsPPORunnerCfg \
     --headless \
     --num_envs=2048
   ```
3. **訓練記錄**
   - 日誌輸出於 `logs/rsl_rl/<experiment_name>/<timestamp_run>/`，保留 `env.yaml`、`agent.yaml` 與 `model_*.pt`。
   - Hydra log 於 `outputs/<date>/<time>/hydra.log`。

## 量化評估基線
1. **TensorBoard 觀察收斂**
   ```bash
   tensorboard --logdir=logs
   ```
   重點指標：`EpisodeReward`, `LinearVelocityTracking`, `Falls`。
2. **離線評測腳本建議**
   - 新增 `scripts/reinforcement_learning/rsl_rl/eval_stairs.py`：固定 policy、關閉學習，跑多回合統計。
   - 指標：平均回報、跌倒率、成功爬樓梯數、能耗、爬升時間。
3. **輸出報表**
   - 將結果存成 CSV，使用 pandas/matplotlib 整理箱型圖或平均 ± 標準差。

## 改良版 PPO 方向
- **網路架構**：
  - 增加 actor/critic 層數或寬度。
  - 引入 LSTM / Transfomer 以利用歷史資訊。
  - 導入 LayerNorm / 自注意力以改善特徵融合。
- **觀測擴充**：
  - 增加階前高度掃描、足端距離、IMU 生資料。
  - 引入多步觀測（時間堆疊），或在 `ObsTerm` 中加入專用特徵工程。
- **動作輸出**：
  - 改為力矩控制 (`JointEffortActionCfg`) 或混合控制。
  - 對 PPO actor 加入 Gaussian policy 網路可學的 `log_std`。
- **演算法參數**：
  - 調整 `clip_param`, `entropy_coef`, `num_learning_epochs`, `desired_kl`。
  - 試驗自適應 entropy/learning rate 調度。
- **Curriculum Learning**：
  - 由低階梯→高階梯逐步提高難度。
  - 加入外部推擠、階梯缺角等擾動增加泛化。

## Lite3 階梯環境（已實作）
- **Gym ID**：`Stairs-Deeprobotics-Lite3-v0`
- **環境設定**：`stairs_env_cfg.py` 基於 rough 版配置，僅保留前進/倒退階梯地形，階梯高度、寬度與摩擦係數會在建置時與 reset 隨機化。
- **新增事件**：`mdp.events.randomize_sim_friction` 每次 reset 重新抽樣靜摩擦與動摩擦，並同步到場景與模擬器；同時關閉外力推擠以專注於樓梯行走。
- **訓練指令**：
   ```bash
   python scripts/reinforcement_learning/rsl_rl/train.py \
      --task=Stairs-Deeprobotics-Lite3-v0 \
      --agent=rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.agents.rsl_rl_ppo_cfg.DeeproboticsLite3StairsPPORunnerCfg \
      --headless \
      --num_envs=2048
   ```
- **評測指令**：可沿用 `play.py` 或自訂 `eval_stairs.py`，記得加入 `--checkpoint` 指向階梯模型並視需求開啟 `--video`。

## 改良版訓練與評測
1. **建立新 config**
   - 在 `agents/` 新增 `DeeproboticsLite3StairsImprovedPPORunnerCfg`。
   - 替換 `policy`、`algorithm` 設定。
2. **重新訓練並保存**
   - 使用相同 `--task`，改 `--agent` 指向新配置。
3. **同一評測流程**
   - 使用相同 eval 腳本、相同種子集。
   - 生成比較報告（baseline vs improved）。

## 比較報告撰寫建議
- **表格**：列出主要指標（平均回報、跌倒率、爬升時間）。
- **圖表**：
  - Reward vs iteration 曲線。
  - 成功率箱型圖。
  - 能耗/穩定性直方圖。
- **錄影**：
  - 使用 `play.py --video` 分別錄製 baseline 與 improved 的示例。
- **結論**：
  - 整理改良帶來的收益、成本（訓練時間、參數數量）與潛在風險。

## 紀錄與追蹤
- 使用 `--logger wandb` 或 `--logger tensorboard` 增加線上追蹤。
- 保存 `env.yaml`/`agent.yaml`，確保可復現。
- 建議建立 `docs/experiments/`，針對每個實驗寫概要（日期、config、重點結論）。

## 附錄
### 常用命令
```bash
# 鍵盤操控 Lite3
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --experiment_name=deeprobotics_lite3_rough \
  --load_run=2025-09-30_18-23-57 \
  --checkpoint=model_19999.pt \
  --keyboard --real-time

# 直接指定 checkpoint 路徑
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Rough-Deeprobotics-Lite3-v0 \
  --checkpoint=/abs/path/to/model.pt \
  --keyboard
```

### 評測腳本骨架（概念）
```python
obs = env.get_observations()
rewards = []
falls = 0
for _ in range(args.num_episodes):
    done = False
    cum_reward = 0.0
    while not done:
        with torch.inference_mode():
            action = policy(obs)
        obs, rew, done, infos = env.step(action)
        cum_reward += rew.mean().item()
        done = done.any()
        if infos.get("termination", {}).get("has_fallen", False):
            falls += 1
    rewards.append(cum_reward)
avg_reward = np.mean(rewards)
fall_rate = falls / args.num_episodes
```

### 後續可擴充的方向
- 自動化評測 CI：在 PR 中執行短程 headless 測試，低於門檻則失敗。
- 導入多機、多 GPU 速度優化 (`--distributed`)。
- 真實機轉移：增加感測延遲、隨機噪聲、摩擦變動等 domain randomization。

## Fork 與維護自己的 GitHub 專案
1. **建立遠端**
   - 在 GitHub 透過 Fork 建立個人倉庫，例如 `yourname/rl_training`。
   - 重新 clone 自己的 fork 或在現有專案加入遠端：
     ```bash
     git remote rename origin upstream
     git remote add origin git@github.com:<yourname>/rl_training.git
     git fetch upstream
     ```
2. **授權與聲明**
   - 保留原始的 `LICENSE`、`LICENSE-robot_lab`、`NOTICE` 等文件。
   - 若新增重大功能，可在 README 或 `NOTICE` 標註「Based on DeepRoboticsLab/rl_training」。
   - 引入第三方資源時，記錄授權於 `docs/licenses.md` 或專案 README。
3. **開發分支策略**
   - `main`：保持可用的穩定版。
   - `feature/*`：針對單一功能或修復建立分支，完成後發 PR 合併到 `main`。
   - 定期同步上游：
     ```bash
     git fetch upstream
     git checkout main
     git merge upstream/main
     git push origin main
     ```
4. **變更紀錄與文件**
   - 每個功能建立對應的 `docs/dev_logs/<date>-<feature>.md`；可沿用 `2025-10-01-stairs-env-sync.md` 範例。
   - 更新 `CHANGELOG.md` 或 GitHub Releases，紀錄重要版本與差異。
5. **測試與驗證**
   - 在 PR 前執行專案既有測試（如 `compileall`、demo play script、headless 短訓練）。
   - 可於 fork 的 GitHub Actions 中設定基本 CI（lint、單元測試、短版模擬）。
6. **發佈流程**
   - 打標版本：`git tag v0.1.0-stairs && git push origin v0.1.0-stairs`。
   - 在 release notes 中列出增強項目、相依性與授權宣告。

---
如需加入更多任務或延伸評測流程，可在此檔案持續補充，確保團隊成員能快速上手。