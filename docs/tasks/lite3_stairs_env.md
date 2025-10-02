# Deeprobotics Lite3 階梯環境說明

## 環境概要
- **Gym ID**：`Stairs-Deeprobotics-Lite3-v0`
- **位置**：`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/stairs_env_cfg.py`
- **延伸自**：`DeeproboticsLite3RoughEnvCfg`，保留原有感測器、動作與獎勵設定，專注於樓梯地形。
- **地形生成**：僅使用 `pyramid_stairs` 與 `pyramid_stairs_inv` 兩種子地形，且於 reset 前會重新抽樣階梯高度（0.08–0.28 m）與階梯寬度（0.28 m）。
- **摩擦隨機化**：
  - 建置時先抽樣一組靜摩擦/動摩擦並套用到地形材質。
  - `mdp.events.randomize_sim_friction` 在每次 reset 時重新抽樣摩擦係數，並同步至場景與模擬器，縮短 domain gap。
- **其他調整**：
  - 關閉外力推擠事件，專注在樓梯攀爬穩定性。
  - 限制命令空間偏向正向 x 方向行走（`lin_vel_x` ∈ [0.1, 0.9]）。
  - 加強 `feet_slide`、`base_height` 等獎勵權重以抑制滑動並維持躯幹高度。

## 訓練設定
- **PPO Runner**：`DeeproboticsLite3StairsPPORunnerCfg`
  - `num_steps_per_env = 32`
  - `num_learning_epochs = 6`
  - `entropy_coef = 0.008`
  - `experiment_name = "deeprobotics_lite3_stairs"`
- **啟動指令**：
  ```bash
  python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Stairs-Deeprobotics-Lite3-v0 \
    --agent=rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.agents.rsl_rl_ppo_cfg.DeeproboticsLite3StairsPPORunnerCfg \
    --headless \
    --num_envs=2048
  ```
- **輸出**：模型與記錄寫入 `logs/rsl_rl/deeprobotics_lite3_stairs/<timestamp>/`，摩擦係數會同步寫入 `env.extras["terrain"]["friction"]` 以利分析。

## 評測與比較
1. **單次推論**：
   ```bash
   python scripts/reinforcement_learning/rsl_rl/play.py \
     --task=Stairs-Deeprobotics-Lite3-v0 \
     --checkpoint=/path/to/model.pt \
     --keyboard \
     --real-time
   ```
2. **批次評測**：建議基於 `eval_stairs.py`（可由 `play.py` 修改）實作 headless 評測腳本，統計：
   - 成功攀登台階數、平均回報、跌倒率。
   - 平均攀爬時間、能耗、足端滑移距離。
3. **比較流程**：
   - 先以原始 PPO（`DeeproboticsLite3RoughPPORunnerCfg`）在階梯環境訓練，取得 baseline。
   - 再以改良版 PPO（階梯專用設定）訓練，並使用上述指標比較。

## 後續優化建議
- **課程學習**：從低高度階梯開始，逐漸提升 `step_height_range` 或加入缺階情境。
- **多感測器資訊**：可在 `stairs_env_cfg.py` 中啟用更多高度掃描點或新增深度影像輔助。
- **真實機考量**：
  - 在事件中加入馬達延遲、關節摩擦隨機化，縮小 sim2real 差距。
  - 評估不同摩擦區間對策略穩定性的影響。
- **記錄與可視化**：配合 TensorBoard 或 W&B 紀錄摩擦係數、成功率等曲線，以便快速定位退化原因。

> 若需擴充更多 stair 變體，可在 `stairs_env_cfg.py` 中調整 `new_sub_terrains` 設定，或引入自訂 heightfield/mesh 生成器。
