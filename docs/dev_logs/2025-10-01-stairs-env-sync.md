# Lite3 樓梯環境整理記錄（2025-10-01）

## 架構概覽
- `DeeproboticsLite3StairsEnvCfg` 直接繼承 `DeeproboticsLite3RoughEnvCfg`，保留既有的觀測、獎勵、事件與動作設定。
- 僅針對地形匯入器覆寫：停用 terrain curriculum，縮減網格數量，並限定為雙向階梯子地形，同時調整階梯尺寸與比例。
- 透過呼叫 `disable_zero_weight_rewards()`，維持和 rough 任務一致的獎勵清理流程，避免播放模式解析空的實體集合。

## 開發流程
1. 重新審視 rough 與 stairs 兩份環境設定，找出非必要的差異。
2. 簡化 `stairs_env_cfg.py`，移除與 rough 配置無關的覆寫，只保留地形相關差異。
3. 確認匯入的子地形設定（rows/cols、proportion、step 寬高等）能產生期望的樓梯地形。
4. 透過 `compileall` 檢查語法，確保調整後的設定可被 Isaac Lab 成功載入。

## 修改檔案
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/stairs_env_cfg.py`：
  - 刪除多餘的命令、獎勵、事件與感測器覆寫。
  - 針對地形匯入器調整行數、列數、子地形清單與比例，形成固定難度的階梯地形。
  - 保留 `disable_zero_weight_rewards()` 呼叫以對齊 rough 配置。

## 驗證指令
```bash
python -m compileall source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/stairs_env_cfg.py
```
