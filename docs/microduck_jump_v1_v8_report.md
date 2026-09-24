# Microduck Jump V1–V8 策略、奖励与训练结果报告

更新日期：2026-09-14

## 1. 报告范围与结论

本报告依据仓库中实际保存的环境配置、训练状态、评估 JSON、TensorBoard 指标和 ONNX 验证记录，汇总 Jump V1–V8。截图中的 `jump-v3` 至 `jump-v8` 是版本化产物目录；V1 的产物位于 `logs/rsl_rl/microduck_jump`，V2 位于 `logs/rsl_rl/jump_v2_transfer` 和 `artifacts/jump_v2_transfer`。

这里严格区分三类结果：

- **训练内成功**：满足该版本当时的内部条件，不等于后续版本的严格验收。
- **评估成功**：通过指定 checkpoint 的固定评估脚本，但不同版本协议不完全相同。
- **最终验收通过**：满足该版本全部高度、姿态、恢复、连续请求、CPU BAM 和导出条件。

截至本报告日期，**V1–V8 没有一个版本完成最终端到端验收并成为可部署策略**。演进中取得的最好阶段性结果是：

- V3 首次学会可靠的名义单跳，旧协议评估 100/100 成功，但 3 cm 阶段实际允许 2.5 cm，且完整稳定性验收未完成。
- V5/V6 在旧单请求协议下达到 100% 高度成功，但漂移、航向、倾斜或姿态超标。
- V7 首次训练连续三跳与 5 秒恢复；最佳零安全失败 checkpoint 的单次跳跃成功率为 81.3%，完整三跳序列率为 44%。
- V8 增加严格脚掌落平、逐腿恢复和拒绝请求惩罚，但当前训练发生策略坍塌，随后被 Warp 编译故障中断；当前 checkpoint 不可用。

## 2. 版本总览

| 版本 | 核心策略变化 | 最终/选定结果 | 最终状态 |
|---|---|---|---|
| V1 | 从零训练；以前沿高度、起跳速度和落地成功奖励探索跳跃 | 高度项有进展，但稳定成功奖励几乎为零，频繁摔倒 | 未通过 |
| V2 | 从 official standing actor 迁移；加入站立教师模仿 | 两次正式训练到 750 update，100 次评估均无起跳 | `diagnose` |
| V3 | 请求式状态机、课程高度、事件型奖励 | 名义单跳 100%，CPU BAM 3/3；完整站立/DR/多 seed 验收因基础设施问题未完成 | `infrastructure_failure` |
| V4 | protocol 2；航向、yaw-rate、头部姿态课程 | 最终单跳成功 65%，航向 P90 17.65°，头部动态偏差约 60° | `needs_stage4_quality` |
| V5 | 精确 3 cm 高度门槛；降低姿态税并诊断航向 | 最终单跳成功 100%，但漂移 P90 6.09 cm、航向 P90 12.29° | `quality_diagnosis_heading` |
| V6 | 加强垂直性、位移、速度、直立、对称和镜像损失 | 最终成功 100%，漂移降到 1.78 cm，但航向 P90 12.47°、恢复倾斜 P90 6.97° | `quality_diagnosis` |
| V7 | 固定 3 cm；单跳/三跳；严格 ready；5 秒恢复 | 最佳零安全失败：单跳 60%，三跳单次 81.3%，完整序列 44% | `budget_exhausted` |
| V8 | 32 秒；恢复态启动；脚掌落平；逐腿 HOME；拒绝与恢复超时 | 当前选定分支成功率 0，出现摔倒/invalid，下一块训练因 Warp 编译错误中断 | `rollback_best` / 中断 |

> 注：V3–V6 的“成功率”采用当时较宽松的单请求恢复协议，不能与 V7/V8 的连续 5 秒恢复和三跳序列成功率直接横向比较。

## 3. 全版本共同基础

### 3.1 控制与接口

- 控制频率：50 Hz。
- Actor 输入：61 维，保持全策略族热切换接口。
- 动作输出：14 维，对应 14 个 Dynamixel XL330 舵机。
- 执行器：BAM 电压控制模型。
- V2–V8 从 official standing actor 初始化，并在包含 standing 样本时使用冻结教师辅助损失：

  `L = L_PPO + 0.1 × MSE(actor_obs, standing_teacher_obs)`

  这是 PPO 优化器的辅助损失，不是环境奖励。

### 3.2 奖励记号

- 表中权重为环境配置中的标量。
- 正权重通常对应正奖励；负权重通常对应非负成本。
- `head_bias` 函数自身返回非正惩罚，因此它虽然使用正权重，效果仍是惩罚头部偏差。
- V3 以后的一次性事件奖励按 `event / dt` 返回，以抵消奖励管理器的 `dt` 缩放；表中仍列其直观事件质量。

### 3.3 通用正则项

各版除跳跃任务奖励外均保留不同强度的通用正则项。它们不是版本创新的主体，但会影响总回报：

| 项目 | V1 | V2 | V3–V8 的典型值 | 作用 |
|---|---:|---:|---:|---|
| body angular velocity | -0.05 | -0.05 | -0.01 左右或由版本专用 upright/yaw 项替代 | 抑制躯干角速度 |
| angular momentum | -0.02 | -0.02 | -0.005 左右 | 抑制全身角动量 |
| joint position limits | -1.0 | -1.0 | -1.0 | 避免关节接近硬限位 |
| self collision | -1.0 | -1.0 | -1.0 | 惩罚自碰撞 |
| body impact | -0.01 | -0.01 | -0.01 | 惩罚非脚部身体撞击 |
| joint torque | -0.001 | -0.001 | -0.001 | 抑制过大力矩 |
| joint torque rate | 课程引入至 -0.05 | 0 | 通常为 0 | 平滑力矩变化 |

后续各版本表格列出跳跃专用奖励及明确改变的 action-rate 等项目。

## 4. V1：从零探索的单次跳跃

### 4.1 策略设计

V1 是最早的 4 秒 episodic 配方，没有请求/ready 状态机，也没有站立策略迁移。训练目标先从站立高度增加 1.5 cm，再切换到增加 3 cm。稳定落地只要求保持 0.25 秒，允许最大 15° 倾斜。

高度奖励采用“历史前沿”设计：只有首次达到更高 trunk z 时才获得增量，原地维持高度不会持续刷分。成功奖励是高度、直立和 HOME 姿态的组合。

### 4.2 奖励函数

| 奖励项 | 权重/课程 | 含义 |
|---|---:|---|
| `jump_height_progress` | +100 | 仅奖励新达到的高度前沿增量 |
| `jump_launch_velocity` | +2 | 奖励受限的向上起跳速度信号 |
| `jump_landing_impact` | -0.02 → -0.05 → -0.10 | 一次性惩罚落地冲击 |
| `jump_success_score` | +10 | 稳定落地后的高度 × 直立 × HOME 姿态一次性奖励 |
| `action_rate_l2` | -0.10 → -0.30 → -0.60 → -1.00 | 后期逐步加强动作平滑 |
| `joint_torque_rate` | 0 → -0.02 → -0.05 | 后期逐步抑制力矩变化 |

### 4.3 最终训练结果

正式运行完成 6000 个 PPO iteration，保存 `model_5999.pt` 和 ONNX。但最终 TensorBoard 指标显示：

- `jump_success_score` 最终为 0，整个训练最大值约 `4.02e-5`，可视为没有学会稳定成功。
- `jump_height_progress` 最终约 0.0207，说明学会了部分向上运动。
- 平均 episode 长度约 101.4 step，即约 2.03 秒，远短于 4 秒上限。
- 最终 `fell_over` 指标约 41.54，摔倒仍是主要终止原因。

结论：V1 学到“制造向上运动”，没有学到“起跳—落地—稳定”的完整技能。高强度且过早引入的 action-rate/torque-rate 税也可能压制动态动作探索。没有保存标准化验收 JSON，ONNX 不应视为已验证部署产物。

## 5. V2：站立策略迁移

### 5.1 策略设计

V2 从 `official/alpha_stand.onnx` 导入 actor，生成 `standing_init.pt`，希望避免 V1 从零同时学习站立与跳跃。episode 延长到 6 秒，约 25% 环境保持站立，75% 环境发出跳跃请求；同时用 0.1 权重的 standing teacher imitation 保护站立能力。

### 5.2 奖励函数

| 奖励项 | 权重 | 含义 |
|---|---:|---|
| `jump_height_progress` | +100 | 高度前沿进展 |
| `jump_launch_velocity` | +2 | 向上速度信号 |
| `jump_launch_drive` | +0.5 | 双脚支撑阶段的起跳驱动力 |
| `jump_landing_impact` | -0.02 | 落地冲击 |
| `jump_success_score` | +10 | 稳定落地后的高度、直立、HOME 组合成功奖励 |
| `action_rate_l2` | 0 | 技能发现阶段不征收动作变化税 |
| standing imitation | 0.1（优化器辅助损失） | 维持 official standing actor 行为 |

### 5.3 最终训练结果

两次正式运行均训练到 750 update，并在 250/500/750 边界进行 nominal 与 DR 评估。各边界 100 个 episode 的结果一致：

- 成功率 0%。
- 起跳率 0%。
- 峰值高度增量 0。

结论：迁移成功保护了站立先验，却没有产生从站立策略到动态起跳的探索桥梁。V2 停在 `diagnose`，没有可部署的跳跃 ONNX。

## 6. V3：请求状态机与课程化事件奖励

### 6.1 策略设计

V3 是第一个真正形成跳跃技能的版本。它引入 protocol 1 请求状态机和显式事件：

- 25% 全程站立，75% 在 1–2 秒发出一次请求。
- 请求前需连续 ready 1 秒。
- 起跳超时 1.5 秒。
- 稳定落地 0.25 秒，随后保持 1 秒判 complete。
- 高度课程依次为 1.5、2.0、3.0 cm。
- 3 cm 阶段的内部通过门槛仍保留 5 mm 容差，即实际 `required_delta=2.5 cm`。
- action-rate 从 0 分阶段增加到 -0.02、-0.05、-0.10。

### 6.2 奖励函数

| 奖励项 | 权重 | 含义 |
|---|---:|---|
| `jump_height_progress` | +1.0 | 归一化高度进展，累计上限约 1 |
| `jump_launch_velocity` | +1.0 | 首次有效离地事件 |
| `jump_launch_progress` | +1.0 | 支撑期向起跳方向的进展，累计上限约 0.1 |
| `jump_landing_impact` | -0.02 | 首次触地冲击 |
| `jump_success_score` | +2.0 | 达到高度并稳定落地 |
| `jump_complete` | +1.0 | 完成完整请求状态机 |
| `jump_failure` | -2.0 | 首次失败事件 |
| `action_rate_l2` | 0 → -0.10 | 技能形成后逐步加入平滑成本 |

一次成功请求的主要正奖励质量上限约为：launch progress 0.1 + takeoff 1 + height 1 + stable success 2 + complete 1。

### 6.3 最终训练结果

选定 checkpoint 为 `014_A_s2/checkpoint_3000.pt`：

- nominal seed 123，100/100 请求成功、100% 起跳、100% 请求接受、0 失败。
- 峰值高度增量 P10 3.279 cm，P90 3.451 cm。
- CPU BAM 30 秒测试 3/3 请求接受并完成，无 invalid，但最大倾斜达到 24.72°。
- ONNX `[1,61]→[1,14]` 验证通过，最大动作误差 `3.81e-6`，normalizer 已烘焙。

不过运行最终标记为 `infrastructure_failure`：一次 250-update 训练块因 segfault 丢失，最终站立、DR 和多 seed 完整验收没有完成。更重要的是，V3 的 3 cm 内部门槛实际是 2.5 cm，且恢复约束远松于 V7/V8。

结论：V3 是“技能发现成功”，不是“最终产品验收成功”。

## 7. V4：航向、yaw-rate 与头部姿态课程

### 7.1 策略设计

V4 升级到 protocol 2，在观测/命令中加入航向误差，并以多个 posture stage 逐步加强航向、yaw-rate 和头部姿态约束。目标原计划从 3 cm 继续升到 3.5/4 cm，但质量门槛未过，因此始终停留在 3 cm。

### 7.2 奖励函数

V4 完整继承 V3 的事件奖励，并增加姿态项：

| 奖励项 | 权重/课程 | 含义 |
|---|---:|---|
| `jump_height_progress` | +1.0 | 归一化高度进展 |
| `jump_launch_velocity` | +1.0 | 有效离地事件 |
| `jump_launch_progress` | +1.0 | 支撑期起跳进展 |
| `jump_landing_impact` | -0.02 | 触地冲击 |
| `jump_success_score` | +2.0 | 稳定落地成功 |
| `jump_complete` | +1.0 | 请求完成 |
| `jump_failure` | -2.0 | 失败事件 |
| `jump_heading` | -0.10 → -0.20 | 航向误差 |
| `jump_yaw_rate` | -0.01 → -0.02 | yaw 角速度 |
| `jump_head_bias` | 0 → +0.5 → +1.0 → +1.5 | 自身为非正项，逐步加强头部偏差惩罚 |
| `action_rate_l2` | -0.02 | 动作平滑 |

### 7.3 最终训练结果

训练 750 个新增 update，最终 checkpoint 为 `block_0750_h30_p4/checkpoint_3000.pt`：

- 单跳成功率 65%，起跳率 100%，请求接受率 100%，安全失败 0。
- 峰值高度 P10 2.446 cm、P90 2.586 cm。
- 漂移 P90 2.689 cm。
- 航向 P90 17.65°，最大 18.83°。
- 头部动态误差 P90 约 59.97°，完成时头部误差约 15.99°。
- 最终 checkpoint 的 CPU BAM 回放接受 3 次请求，但只成功完成 2 次；最大倾斜 8.64°。

结论：姿态税显著改变了优化目标，却没有真正修正头部和航向，反而损害了高度与成功率。状态停在 `needs_stage4_quality`。

## 8. V5：精确 3 cm 与航向诊断

### 8.1 策略设计

V5 将成功高度改为精确 `required_delta=target_delta`，不再保留 V3 的 5 mm 容差；只做 nominal 课程。高度计划为 3/3.5/4/4.5/5 cm，但由于质量门槛始终未稳定通过，最终停在 3 cm。头部惩罚固定为较温和的 +0.5，并尝试过 stronger heading tier。

### 8.2 奖励函数

| 奖励项 | 权重/课程 | 含义 |
|---|---:|---|
| `jump_height_progress` | +1.0 / +1.5 / +2.0，最终 +1.5 | 高度课程的主要进展奖励 |
| `jump_launch_velocity` | +1.0 | 有效离地事件 |
| `jump_launch_progress` | +1.0 | 起跳前进展 |
| `jump_landing_impact` | -0.02 | 落地冲击 |
| `jump_success_score` | +2.0 | 达到精确高度后的稳定落地 |
| `jump_complete` | +1.0 | 请求完成 |
| `jump_failure` | -2.0 | 失败事件 |
| `jump_heading` | -0.20，strong tier -0.30 | 航向误差 |
| `jump_yaw_rate` | -0.02，strong tier -0.03 | yaw 角速度 |
| `jump_head_bias` | +0.5 | 头部 DC 偏差惩罚 |
| `action_rate_l2` | -0.02 | 动作平滑 |

### 8.3 最终训练结果

训练 1500 个新增 update，最终 checkpoint 为 `block_1500_h30/checkpoint_4375.pt`：

- 精确 3 cm 单跳成功率 100%，高度、起跳、请求接受和站立均为 100%，安全失败 0。
- 峰值高度 P10 3.660 cm、P90 3.709 cm。
- 漂移 P90 6.089 cm。
- 航向 P90 12.292°，超过质量门槛。

较早的 `block_0250` 曾达到 100% 成功、漂移 P90 2.873 cm、航向 P90 1.603°，但后续训练发生明显质量回退，且没有形成连续两个边界的稳定通过。

结论：V5 解决了“精确跳够 3 cm”，却没有解决训练后期的漂移和航向退化；状态为 `quality_diagnosis_heading`，未进入 3.5 cm。

## 9. V6：垂直跳精度与镜像约束

### 9.1 策略设计

V6 把目标集中到“垂直、对称、落地姿态接近 HOME”，计划在质量过关后从 3 cm 逐步升到 10 cm。除环境奖励外启用 PPO symmetry mirror loss，系数 0.5。实际仍停留在 3 cm。

### 9.2 奖励函数

| 奖励项 | 权重 | 含义 |
|---|---:|---|
| `jump_height_progress` | +1.5 | 高度进展 |
| `jump_launch_velocity` | +1.0 | 有效离地事件 |
| `jump_launch_progress` | +1.0 | 起跳前进展 |
| `jump_landing_impact` | -0.02 | 落地冲击 |
| `jump_success_score` | +2.0 | 稳定落地成功 |
| `jump_complete` | +1.0 | 请求完成 |
| `jump_failure` | -2.0 | 失败事件 |
| `jump_heading` | -0.40 | 航向误差 |
| `jump_yaw_rate` | -0.04 | yaw 角速度 |
| `jump_head_bias` | +0.5 | 头部偏差惩罚 |
| `action_rate_l2` | -0.02 | 动作平滑 |
| `jump_planar_displacement` | -0.35 | 以 2 cm 为尺度惩罚水平位移 |
| `jump_planar_velocity` | -0.15 | 以 0.2 m/s 为尺度惩罚水平速度 |
| `jump_upright` | -0.12 | 2° deadband、15° cap 的直立成本 |
| `jump_post_landing_pose` | -0.10 | 触地后 HOME 姿态误差，尺度 0.2 rad |
| `jump_launch_action_symmetry` | -0.08 | 起跳期左右动作不对称，尺度 0.2 rad |
| symmetry mirror loss | 0.5（优化器辅助损失） | 约束镜像观测下的左右对称动作 |

### 9.3 最终训练结果

训练 2000 个新增 update，最终 checkpoint 为 `block_02000_h030/checkpoint_6375.pt`：

- 单跳成功、达到高度、起跳、请求接受和站立均为 100%，安全失败 0。
- 峰值高度 P10 3.367 cm、P90 3.469 cm。
- 漂移 P90 1.782 cm，最终漂移 P90 1.746 cm；仍高于 1 cm 门槛。
- 航向 P90 12.467°、最大 13.049°。
- 最终恢复倾斜 P90 6.967°。
- HOME pose 误差 P90 0.158 rad。

`block_01250` 的最终漂移曾低至 0.468 cm、航向 P90 2.419°，但最大航向过程误差 13.314°、倾斜 P90 5.805°，仍未通过整套质量门槛。

结论：位移奖励确实降低了漂移，但多项强约束形成新的折中解；最终状态 `quality_diagnosis`，没有升高。

## 10. V7：连续请求与完整恢复

### 10.1 策略设计

V7 不再追求升高，固定精确 3 cm，把训练重点转向连续跳和落地恢复：

- episode 24 秒。
- 50% 单跳、30% 三跳、20% 全程站立。
- 三跳请求位于 7/14/21 秒并带 ±0.5 秒抖动。
- 再次请求前需连续 ready 2 秒：双脚接触、无身体接触、倾斜 ≤3°、平均 HOME 误差 ≤0.08 rad、水平速度 ≤0.03 m/s、yaw-rate ≤0.1 rad/s。
- 落地后需连续稳定 5 秒才 complete。

V7 删除 V3–V6 的旧奖励堆栈，按阶段分别限制起跳/飞行和触地后恢复。

### 10.2 奖励函数

| 阶段 | 奖励项 | 权重 | 含义 |
|---|---|---:|---|
| 事件 | height progress | +1.5 | 精确 3 cm 高度进展 |
| 事件 | complete | +3.0 | 完成 5 秒稳定恢复 |
| 事件 | failure/body contact | -8.0 | 摔倒或身体触地 |
| 事件 | takeoff timeout | -3.0 | 请求后未及时起跳 |
| 事件 | landing impact | -0.03 | 首次触地冲击 |
| 全程 | action rate | -0.02 | 动作平滑 |
| 全程 | head bias | +0.5 | 头部偏差惩罚 |
| 起跳/飞行 | planar displacement | -0.15 | 水平位移 |
| 起跳/飞行 | planar velocity | -0.10 | 水平速度 |
| 起跳/飞行 | heading | -0.25 | 航向误差 |
| 起跳/飞行 | yaw-rate | -0.03 | yaw 角速度 |
| 起跳/飞行 | upright | -0.08 | 躯干倾斜 |
| 起跳/飞行 | action symmetry | -0.08 | 左右动作不对称 |
| 触地/恢复 | planar displacement | -0.03 | 弱化的持续位置成本 |
| 触地/恢复 | planar velocity | -0.25 | 强化静止 |
| 触地/恢复 | heading | -0.05 | 恢复航向 |
| 触地/恢复 | yaw-rate | -0.08 | 抑制持续旋转 |
| 触地/恢复 | upright | -0.40 | 强化直立 |
| 触地/恢复 | HOME pose | -0.25 | 平均关节 HOME 误差 |
| 触地/恢复 | single-foot support | -0.25 | 惩罚单脚支撑 |

### 10.3 最终训练结果

总预算 3000 个新增 update 用尽，最终 checkpoint 并不是最佳 checkpoint：

**最终 checkpoint** `v5_block_2495_ledger_3000/checkpoint_6870.pt`：

- 单跳成功率 53%。
- 三跳单次成功率 70.33%，完整三跳序列率 31%。
- 请求接受率 100%，安全失败 0。
- 三跳漂移 P90 9.87 cm，恢复倾斜 P90 7.24°。

**最佳零安全失败 checkpoint** `v5_block_2000_ledger_2505/checkpoint_6375.pt`：

- 单跳成功率 60%。
- 三跳单次成功率 81.33%，完整三跳序列率 44%。
- 请求接受率 100%，摔倒、身体触地、invalid 均为 0。
- 单跳/三跳高度 P10 分别约 2.915/2.944 cm。
- 单跳漂移 P90 3.023 cm；三跳漂移 P90 9.323 cm。
- 单跳/三跳恢复倾斜 P90 分别约 6.47°/6.77°。

结论：V7 首次证明策略可以安全接受连续请求，但 24 秒 episode 给 21 秒第三跳留下的恢复窗口不足；平均 HOME 指标也掩盖了单腿、单踝和脚跟抬起。状态为 `budget_exhausted`，未导出验收策略。

## 11. V8：脚掌落平与连续恢复

### 11.1 策略设计

V8 针对 V7 暴露的四个结构问题重构：

- episode 从 24 秒延长到 32 秒，使第三跳获得完整恢复窗口。
- 请求到达但未 ready 时显式记失败，禁止通过拒绝请求逃避任务。
- ready 检查增加任一腿关节、左右踝以及双脚掌 pitch/roll 约束。
- 恢复成本延续到下一次请求前的 `settle_hold`，避免 complete 后再次漂离。
- 加入恢复态程序化初始分布，直接覆盖倾斜、速度、关节和单侧髋/踝偏置。
- 训练高度 potential 指向 3.2 cm，成功和对外验收仍要求精确 3.0 cm。

严格 ready 需要连续 2 秒满足：双脚接触、无身体接触、倾斜 ≤3°、平均 HOME ≤0.08 rad、任一腿关节 ≤0.10 rad、任一踝 ≤0.06 rad、脚掌 pitch ≤3°、脚掌相对 HOME roll 误差 ≤3°、水平速度 ≤0.03 m/s、yaw-rate ≤0.1 rad/s、roll/pitch 角速度范数 ≤0.5 rad/s。

触地后稳定 5 秒 complete；8 秒仍未完成触发 recovery timeout，并解除 busy。

### 11.2 奖励函数

| 阶段 | 奖励项 | 权重 | 含义 |
|---|---|---:|---|
| 事件 | height progress | +1.5 | 面向 3.2 cm potential 的高度进展 |
| 事件 | complete | +4.0 | 5 秒完整恢复 |
| 连续 | recovery potential progress | +1.5 | 双脚、倾斜、速度、HOME 和脚掌综合 potential 的增量 |
| 事件 | fall/body contact | -8.0 | 摔倒或身体触地 |
| 事件 | takeoff timeout | -3.0 | 起跳超时 |
| 事件 | recovery timeout | -4.0 | 8 秒内未完成恢复 |
| 事件 | readiness rejection | -4.0 | 请求时未达到 ready |
| 事件 | landing impact | -0.03 | 触地冲击 |
| 全程 | action rate | -0.02 | 动作平滑 |
| 全程 | head bias | +0.5 | 头部偏差惩罚 |
| 起跳/飞行 | planar displacement | -0.20 | 水平位移 |
| 起跳/飞行 | planar velocity | -0.10 | 水平速度 |
| 起跳/飞行 | heading | -0.25 | 航向误差 |
| 起跳/飞行 | yaw-rate | -0.03 | yaw 角速度 |
| 起跳/飞行 | upright | -0.08 | 倾斜 |
| 起跳/飞行 | action symmetry | -0.10 | 左右动作不对称 |
| 触地事件 | touchdown displacement | -0.50 | 以 2 cm 为尺度，一次性惩罚落地点漂移 |
| `settle_hold` | planar velocity | -0.25 | 恢复并保持静止 |
| `settle_hold` | yaw-rate | -0.08 | 抑制旋转 |
| `settle_hold` | upright | -0.40 | 保持直立 |
| `settle_hold` | robust leg HOME | -0.35 | 腿关节 p4 聚合并单算左右踝 |
| `settle_hold` | foot pose | -0.40 | 脚掌 pitch/roll 姿态 |
| `settle_hold` | single-foot support | -0.30 | 单脚支撑 |
| `settle_hold` | absolute heading | -0.02 | 防止完成后慢速偏航 |

### 11.3 当前训练结果

V8 先完成 64 环境 × 5 update smoke，再从两个 V7 checkpoint 分支训练：A 500 update，B 1000 update；累计新增 update 为 1505。当前选择 B 分支并将学习率降到 `5e-5`，状态记录为 `rollback_best`。

当前/最近评估并不理想：

- A 分支 250 update：单跳、三跳成功率均为 0；单跳请求接受 100%，但摔倒/invalid 100%，恢复倾斜约 53°。
- A 分支 500 update：仍为 0 成功；单跳接受率降到 38%，readiness failure 62%。
- B 分支 1000 update：单跳、三跳成功率仍为 0，峰值高度为 0；摔倒/invalid 100%。单跳请求接受 100%，三跳仅第一跳被接受，整体接受率约 33.3%。
- 下一块 B1250 训练在 Warp codegen 阶段因 `Value after * must be iterable, not function` 中断。

此外，当前实现审计发现三个会干扰 V8 学习或评估的问题：

1. recovery-only 初始状态的 controller 没有真实起跳峰值，但完成逻辑仍依赖高度成功，导致恢复样本可能无法完成。
2. 部分脚掌/HOME 平方成本缺少稳健限幅，在大偏差摔倒态可能产生远大于事件奖励的负回报，形成梯度支配。
3. 严格评估器尚未完整传递逐关节、逐踝和脚掌指标，导致 supervisor 的模型选择依据不完整。

结论：V8 的问题定义比 V7 更正确，但当前实现和课程还没有建立有效学习信号。现有 checkpoint 不可用，也尚未消耗完计划的 4000-update 总预算；在修复实现问题和 Warp 编译故障前，不应直接继续长训。

## 12. 跨版本指标对比

| 版本 | 评估协议 | 单跳成功 | 三跳单次成功 | 完整三跳 | 主要失败指标 |
|---|---|---:|---:|---:|---|
| V1 | 4 秒、无请求状态机 | 近似 0 | — | — | 成功奖励几乎为 0，频繁摔倒 |
| V2 | 6 秒、单请求 | 0% | — | — | 起跳率 0% |
| V3 | 旧协议、2.5 cm 内部门槛 | 100% | — | — | CPU 最大倾斜 24.72°，完整验收中断 |
| V4 | 旧协议、姿态课程 | 65% | — | — | 航向 P90 17.65°，头部误差大 |
| V5 | 旧协议、精确 3 cm | 100% | — | — | 漂移 P90 6.09 cm，航向 P90 12.29° |
| V6 | 旧协议、精确 3 cm | 100% | — | — | 漂移 P90 1.78 cm，倾斜 P90 6.97° |
| V7 最佳 | 5 秒恢复、单跳/三跳 | 60% | 81.33% | 44% | 三跳漂移 P90 9.32 cm，倾斜 P90 6.77° |
| V8 当前 | 严格 ready、恢复态混合 | 0% | 0% | 0% | 摔倒/invalid 100%，训练编译中断 |

## 13. 演进分析

### 13.1 哪些变化有效

- V2 的 standing actor 迁移保留了站立基础，成为后续版本的合理初始化方式。
- V3 的请求状态机、事件奖励和分阶段 action-rate 首次解决了技能发现问题。
- V5 的精确高度门槛消除了“标称 3 cm、实际只需 2.5 cm”的验收漏洞。
- V6 的位移和垂直性成本显著降低了单跳漂移，证明方向正确。
- V7 的连续请求数据揭示了单跳评估隐藏的问题：策略可以完成一次跳，却不能可靠回到下一次起跳状态。
- V8 的逐腿、逐踝、脚掌落平、拒绝事件和 `settle_hold` 在任务定义层面直接针对 V7 的真实失效模式。

### 13.2 为什么总体效果仍不理想

- **目标不断加严，但课程没有重新建立可学习的桥梁。** V4、V6 和 V8 一次加入多项强约束后，旧策略获得的正奖励骤减，负成本支配优势估计。
- **旧协议成功率掩盖恢复缺陷。** V3–V6 只需较短稳定窗口；V7 改成 5 秒并要求再次 ready 后，真实连续成功率显著下降。
- **平均指标掩盖局部结构问题。** V7 的 14 关节平均 HOME 误差允许单侧髋或踝有明显偏差，脚仍有接触时也可能脚跟抬起。
- **episode 时长和请求安排不匹配。** V7 的第三跳在 21 秒发出，而 episode 在 24 秒结束，天然无法观察完整 5 秒恢复。
- **位置恢复奖励存在行为折中。** 持续“回原点”会鼓励触地后走动，可能改善最终位置却破坏 ready；V8 改成触地一次性漂移成本是合理修正。
- **V8 实现与监督链尚未闭合。** recovery-only 完成条件、无界成本、严格指标传递和 Warp 编译错误会使正确的设计无法转化为有效训练。

## 14. 建议的后续顺序

在继续任何长训前，建议按以下顺序处理：

1. 修复 V8 recovery-only 完成语义，使恢复样本只优化恢复，不依赖虚假的跳高峰值。
2. 对脚掌、HOME 和摔倒态成本做有界/稳健化，并用实际 episode reward mass 验证所有 penalty 日志均非正且量级可控。
3. 修复严格 evaluator/supervisor 的逐腿、逐踝、脚掌、ready 延迟和 busy-stuck 指标传递。
4. 定位 Warp `Value after * must be iterable` 的具体 kernel/config 参数，先恢复 64×5 smoke。
5. 增加小规模恢复态单元训练/评估，确认 recovery potential 静止收益为零、改善为正、恶化为负。
6. 从 V7 两个候选 checkpoint 重新做短 A/B；初期降低 recovery-only 比例或分阶段加入最强脚掌成本，避免一次性改变整个优势分布。
7. 只有连续两个 250-update 边界通过中间门槛后，才进入以三跳为主的巩固分布。
8. 最终使用全局最佳而非最后 checkpoint，完成 seeds 123/124/125、CPU BAM 五跳和 normalizer-baked ONNX 验证后再判定可部署。

## 15. 证据索引

- V1 正式训练：`logs/rsl_rl/microduck_jump/2026-09-05_13-43-02_jump_v1/`
- V2 设计与结果：[microduck_jump_policy.md](microduck_jump_policy.md)
- V2 正式训练：`logs/rsl_rl/jump_v2_transfer/2026-09-05_17-57-59_train/`、`2026-09-05_21-57-42_train/`
- V3 设计：[microduck_jump_v3.md](microduck_jump_v3.md)
- V3 状态：`artifacts/jump-v3/status.json`
- V3 nominal 评估：`artifacts/jump-v3/evaluations/014_A_s2_nominal_123.json`
- V3 CPU BAM：`artifacts/jump-v3/delivery/cpu_bam.json`
- V4 状态与评估：`artifacts/jump-v4/run/state.json`、`artifacts/jump-v4/evaluations/`
- V5 状态与评估：`artifacts/jump-v5/run/state.json`、`artifacts/jump-v5/evaluations/`
- V6 状态与评估：`artifacts/jump-v6/run/state.json`、`artifacts/jump-v6/evaluations/`
- V7 状态与评估：`artifacts/jump-v7/run/state.json`、`artifacts/jump-v7/evaluations/`
- V8 状态与评估：`artifacts/jump-v8/run/state.json`、`artifacts/jump-v8/evaluations/`
- 版本课程与权重实现：`src/mjlab_microduck/jump_curriculum.py`
- controller、事件和奖励实现：`src/mjlab_microduck/tasks/mdp.py`
- 迁移 PPO/教师辅助损失：`src/mjlab_microduck/jump_runner.py`
