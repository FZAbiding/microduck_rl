# Microduck 跳高策略说明

任务 ID 是 `Mjlab-Jump-Flat-MicroDuck`；带齿隙模型的 A/B 版本是
`Mjlab-Jump-Flat-Backlash-MicroDuck`。策略接口和 walking 家族一致：

- actor observation 是 61D：48D 本体状态，后接
  `[twist(3), head_pose(4), body_pose(6)]`；
- action 是 14D，对应 14 个 XL330 舵机；
- critic 可以看到额外的线速度和脚部接触信息，actor 不会因此改变接口。

## 奖励和 episode

一个 episode 最长 6 秒。策略从 HOME 附近开始，奖励只鼓励飞行期间新增的躯干高度，并在训练目标处封顶；实测峰值始终单独保存，保持高度不会产生 jackpot。收到请求前必须双脚稳定支撑，双脚连续离地两个 control step 才会锁存起跳；1.5 秒仍未起跳则请求超时，起跳后的第一次触地才计算一次冲击。落地后双脚接触、躯干高度、倾角、垂直速度、水平角速度和非脚部接触都满足要求并持续 0.25 秒，才支付一次成功奖励；稳定站立累计 1 秒后才重新接受按键。

“会跳”要看起跳率、峰值高度和落地率；“安全落地”要同时看 stable-landing-rate、fall-rate、body contact 和 landing impact。不要只看总 reward。

## PPO 日志怎么读

iteration 不是秒数；本任务每个环境每 iteration 推进 24 个 control steps。PPO 的 value loss、policy loss 和 KL 是优化器稳定性的诊断量，不是“越低越好”的任务目标：loss 可能因奖励尺度和课程阶段变化而上升，行为仍然可能改善。首先检查它们有限、没有爆炸，再以跳跃行为指标为准。

## 训练、评估和导出

训练前先做 3 秒 HOME settle check；它会检查全程最大倾角、非脚部接触和最终高度，
因此失败时应先修正 HOME 控制/模型平衡，再解释 RL 指标：

```bash
uv run scripts/settle_jump.py --seconds 3 --num-envs 64 --joint-noise 0.03
```

先做便宜的 smoke test：

```bash
uv run train Mjlab-Jump-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.init-checkpoint artifacts/jump_v2_transfer/standing_init.pt \
  --agent.save-interval 1 --agent.logger tensorboard \
  --agent.upload-model False --enable-nan-guard True
```

正式训练由每 250 iterations 的真实回放评估控制，最多 6000 iterations。每个阶段只有连续三个 checkpoint 达标才升级；连续三个评估点无改善会停止并保存诊断回放：

```bash
uv run scripts/train_jump_transfer.py \
  --num-envs 4096 --max-iterations 6000 \
  --init-checkpoint artifacts/jump_v2_transfer/standing_init.pt
```

用实际 checkpoint 做无 GUI 评估：

```bash
uv run scripts/evaluate_jump.py \
  --checkpoint-file logs/rsl_rl/jump_v2_transfer/<run>/model_XXXX.pt \
  --episodes 100 --mode nominal --json-out artifacts/jump_eval_nominal.json \
  --trace-out artifacts/jump_eval_nominal.npz
```

`--mode dr` 保留 checkpoint 对应课程阶段的域随机化（也可用 `--stage 2/3` 显式指定）。只有连续三个评估 checkpoint 同时达到 nominal
成功率和稳定落地率至少 80%、DR 成功率至少 70%、峰值高度达标率至少 80%，并且
fall/NaN/严重 body impact 为零，才把它视为收敛。

导出必须走带 normalizer 的仓库路径：

```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/jump_v2_transfer/<run>/model_XXXX.pt \
  --onnx-file artifacts/jump/jump.onnx --num-envs 1
```

导出的 ONNX 应是 `61 -> 14`，然后再用
`scripts/infer_policy.py` 做至少 5 秒的 CPU BAM rehearsal。


官方站立策略的迁移起点由 `scripts/import_standing.py` 生成。它保存 actor、冻结
teacher、实际 ONNX 归一化分母和源文件 hash；不保存官方 PPO optimizer 或 iteration，
因此 `--agent.init-checkpoint` 是新任务起点，`--resume` 才是完整训练续接。

按键演示使用同一个 jump actor。J 发送一次请求，P 只做单独的物理推扰测试；无 GUI
且可复现的 CPU BAM 回放例如（将路径替换为通过验收的 ONNX）：

```bash
uv run scripts/infer_policy.py --jump <accepted-jump.onnx> \
  --headless --seconds 10 --jump-at 1.5 5.0 \
  --json-out artifacts/jump_v2_transfer/cpu_jump.json \
  --video-out artifacts/jump_v2_transfer/cpu_jump.mp4
```

验收要求是 nominal 100 回合完整成功率至少 80%、训练随机化配置至少 70%，且连续
三个 checkpoint 达标。最新迁移训练在第 750 iteration 的三次评估仍均为零起跳，
已按规则停止为 `diagnose`；当前没有可部署的 `jump.onnx`，候选模型和未剪辑诊断回放保存在
`logs/rsl_rl/jump_v2_transfer/2026-09-05_21-57-42_train/`，不能当作已收敛的
`jump.onnx` 使用。


## Jump-V7：3 cm 落地恢复基础阶段

V7 固定 3 cm，不进入升高课程。训练回合为 24 秒，采样分布是 50% 单跳、
30% 在约 7/14/21 秒发出三次请求、20% 全程站立；每个请求时刻独立加入
±0.5 秒抖动。第一次请求允许从兼容 V5/V6 的安全站姿启动；接受过第一次请求后，
再次起跳必须连续 2 秒满足双脚接触、无身体接触、倾斜不超过 3°、HOME 关节
平均 L1 不超过 0.08 rad、水平速度不超过 0.03 m/s、yaw-rate 不超过
0.1 rad/s。

V7 只有落地后连续稳定 5 秒才支付一次完成奖励。评估器不会在该事件发生时提前结束，
而会继续观察到 15 秒单跳或 30 秒三跳回合结束，从而捕捉“完成后再摔倒”和第三跳
累积失稳。V7 是 nominal-only 基础阶段；通过前不得升到 4 cm。

完整双分支监督训练（64×5 smoke 也计入 3000-update 总账）：

```bash
uv run scripts/supervise_jump_v7.py \
  --v5-checkpoint artifacts/jump-v5/block_1500_h30/checkpoint_4375.pt \
  --v6-checkpoint artifacts/jump-v6/block_01250_h030/checkpoint_5625.pt \
  --output artifacts/jump-v7 --num-envs 4096 --device cuda:0
```

单独复现固定 seed 123 的两个验收电池：

```bash
uv run scripts/evaluate_jump.py --checkpoint-file <v7.pt> \
  --episodes 100 --seed 123 --sequence single \
  --json-out artifacts/jump-v7/single-123.json

uv run scripts/evaluate_jump.py --checkpoint-file <v7.pt> \
  --episodes 100 --seed 123 --sequence triple \
  --json-out artifacts/jump-v7/triple-123.json
```

监督器会在连续两个 250-update 边界都通过后，自动复测 seeds 123/124/125，
再走标准 normalizer-baked ONNX 导出、`[1,61] -> [1,14]` 误差检查和 30 秒
CPU BAM 7/14/21 秒三跳回放。只有这些门全部通过，状态才会写为 `passed`。
