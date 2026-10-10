# Microduck Jump V9 策略、奖励与训练结果报告

更新日期：2026-10-11。实验执行于 2026-09-25 与 2026-10-08；代码基于 `6b4d385`。

## 1. 报告范围与结论

本报告延续 [V1–V8 报告](microduck_jump_v1_v8_report.md) 的结构，依据本地保存的
配置、checkpoint、严格评估 JSON、训练奖励积分、ONNX 校验及 CPU BAM 回放。
详细过程见 [执行记录](microduck_jump_v9_execution.md)，原始设计见
[训练前计划](microduck_jump_v9_plan.md)。计划中的阶段与验收目标不代表已经完成。

**V9 未通过最终验收，连续跳跃效果仍不理想。** 本轮主要成果是修复 V8 的测量、
恢复状态机、奖励单位及训练监督问题，并确认探索噪声阻断严格 ready 数据的原因。

- 同协议最佳诊断候选 A6425：站立 100%，近 HOME 合成恢复 100%，严格单跳 22%，
  完整三跳 13%；仅 seed 123、每项 100 回合，不能宣称训练稳定或实机可用。
- idle 探索噪声降至原值 0.25 倍后，训练开始产生跳高与完成奖励，但追加 50 update
  的 A6475 单跳降至 9%、完整三跳降至 0%，确认复测后回滚。
- 用户实际 CPU BAM 回放及独立复现中，A6425 接受第 7 秒请求，拒绝第 17/27 秒请求；
  腿关节最大误差持续略超 0.10 rad，无法重新 ready。
- 最终状态 `diagnose`；A 活动指针恢复到 A6425，下一次试验 LR 2.5e-5。
  B 保留 V7 B6125 父模型。累计消耗与预留均为 165/4000 update，无后台训练。

## 2. 策略设计与工程修复

### 2.1 保持的物理与部署接口

- 50 Hz 控制、BAM 电压执行器、61D actor 输入、14D 动作；无动作 EMA。
- 61D 输入保持 48D 本体感知与 13D 命令块，跳跃请求位于 twist vx 槽。
- 从 V7 A6375/B6125 初始化，保留 actor、normalizer、critic 与 optimizer；
  加载后显式覆盖并断言实际学习率，而非仅修改配置文件。
- 标准 ONNX 导出烘焙 normalizer；CPU BAM、GPU 评估共享 V9 姿态测量定义。

### 2.2 V9 修复内容

| 问题 | V9 处理 |
|---|---|
| 纯恢复样本被跳高条件阻断 | 独立 recovery-complete 事件，不伪造起跳高度，不安排请求 |
| reset 与命令重采样类别不一致 | 同一次 episode 分类供物理 reset 和命令使用 |
| 右踝索引错误、腿误差混入头部 | 按舵机关节名称解析双腿和双踝 |
| 脚 site 选择器未解析，选中所有 site | 显式选择 left_foot/right_foot，加入干扰 site 回归测试 |
| 评估缺少严格姿态输入却补零 | V9 缺必填测量直接报错；未观察事件用 null |
| 拒绝事件在 manager 调用间丢失 | 排队并消费一次 |
| 超时后仍补发完成奖励 | V9 超时完成无效；多请求累计超时 |
| 脚/腿成本失控、触地成本被 dt 缩小 | V9 成本限幅，触地事件补偿 dt，pilot 低权重起步 |
| 回滚指针和 worker 父模型不一致 | 单一续训指针、父 SHA 校验、实际 LR 与预算记录 |
| 探索噪声阻断 ready | 增加绝对 idle-std 倍率，加载后设置，续训不重复累乘 |

V3–V8 历史奖励与测量路径保留；新行为由 policy_version=9 启用。
脚 roll 当前为超过 HOME 约 5° 标定幅度的部分，pitch 对地面测量。
尚未增加足跟/足尖支撑几何验收，不能把角度代理量当作完整脚掌接触证明。

### 2.3 请求与恢复契约

首次及后续请求均要求连续 2 s 满足：双脚接触、无身体触地、站立高度误差
≤1 cm、倾斜 ≤3°、竖直速度 ≤0.05 m/s、水平速度 ≤0.03 m/s、
躯干角速度范数 ≤0.5 rad/s、yaw-rate ≤0.1 rad/s、腿平均误差 ≤0.08 rad、
腿最大误差 ≤0.10 rad、踝最大误差 ≤0.06 rad、脚 pitch/roll 代理误差 ≤3°。

有效离地与高度 ≥3.0 cm 后，落地连续稳定 5 s 才计跳跃完成；恢复超过 8 s
记超时并释放 busy，但这不会绕过下一次请求的 ready 门槛。
CPU 回放中的 `ignored (not ready)` 同时涵盖 busy 和 ready 不足，不能仅凭该行区分。

## 3. 奖励与训练配置

### 3.1 实际 pilot 奖励

下表为本轮执行配置，而非尚未执行的后续阶段权重。事件通过 dt 补偿保持一次性质量，
连续成本按时间积分；recovery potential 为有符号进展。

| 阶段 | 奖励项 | pilot 权重 | 含义 |
|---|---|---:|---|
| 高度进展 | jump_height_progress | +1.5 | shaping 指向 3.2 cm，验收仍为 3.0 cm |
| 完成事件 | jump_complete | +4.0 | 跳高合格且完整恢复 |
| 恢复进展 | jump_recovery_potential | +1.5 | 综合恢复势函数增量 |
| 完成事件 | jump_recovery_complete | +4.0 | 纯恢复独立完成，不计跳跃成功 |
| 失败事件 | jump_failure | -8.0 | 摔倒/身体触地等失败 |
| 超时事件 | jump_timeout | -3.0 | 起跳超时 |
| 超时/拒绝事件 | jump_recovery_timeout / jump_readiness_failure | -4.0 / -4.0 | 恢复超时或请求被拒绝 |
| 触地事件 | jump_landing_impact / jump_touchdown_displacement | -0.03 / -0.125 | 冲击和落地点漂移 |
| 起跳/飞行 | planar displacement / velocity | -0.20 / -0.10 | 抑制水平运动 |
| 起跳/飞行 | heading / yaw-rate / upright | -0.25 / -0.03 / -0.08 | 航向与姿态 |
| 起跳/飞行 | action symmetry | -0.10 | 动作对称 |
| 恢复保持 | velocity / yaw-rate | -0.25 / -0.08 | 停稳 |
| 恢复保持 | upright / leg HOME / foot pose | -0.10 / -0.0875 / -0.10 | 直立、腿恢复、脚掌姿态 |
| 恢复保持 | single-foot / heading | -0.075 / -0.02 | 双脚支撑与航向 |
| 全程 | action_rate_l2 | -0.02 | 平滑动作 |
| 全程 | jump_head_bias | +0.5 | 函数自身为非正，正权重仍是惩罚 |

通用正则保留 body angular velocity -0.01、angular momentum -0.005、
关节限位 -1、自碰撞 -1、身体冲击 -0.01、扭矩平方 -0.001，扭矩变化率为 0。

### 3.2 采样与探索

实际只进入 nominal pilot：standing/recovery/single/triple = 30%/10%/60%/0%，
恢复初态为低噪声近 HOME 合成状态，recovery noise scale=0.25；该值与动作探索
idle-std scale 是两个不同参数。训练回合 42 s，pilot 请求候选时刻 7/17/27 s、
±0.25 s 抖动；single 样本只使用首个请求。三跳训练课程尚未执行。

正式块为 4096 env × 24 step/update，seed 42，实际 LR 5e-5，带 standing
教师辅助损失和镜像设置。A 请求期沿用可学习探索 std；idle 原始非头部 std=0.05、
头颈=0.03。低噪声块改为 0.0125/0.0075，请求期间探索分布不变。

## 4. 训练过程与预算

| 实验 | 父模型 | 环境数 | 新增 update | 用途/结果 |
|---|---|---:|---:|---|
| smoke-001/002 | 见各配置 | 各 64 | 5+5 | 工程冒烟及超时奖励修订复测 |
| A-pilot-0050 | V7 A6375 | 4096 | 50 | 原噪声，保留诊断候选 A6425 |
| B-pilot-0050 | V7 B6125 | 4096 | 50 | 站立退化，回滚 B6125 |
| smoke-003 | V9 A6425 | 64 | 5 | 0.25 倍 idle 噪声与导出验证 |
| A-pilot-0100 | V9 A6425 | 4096 | 50 | 低噪声候选 A6475 退化，回滚 |
| 合计 | | | **165** | 占 4000-update 上限，未耗尽预算 |

checkpoint 文件名中的 6425/6475 包含祖先更新，不能当作 V9 本轮消耗。
零更新诊断和行为评估不计 PPO update；失败尝试的预留不会因回滚消失。

## 5. 严格评估结果

nominal、seed 123、每项 100 回合；单跳 20 s，第 7 s 请求；三跳 40 s，
第 7/14/21 s 固定请求，不等待 ready 自动推迟。成功分母包含拒绝和未完成请求。

| 候选 | standing | 合成 recovery | single | 完整 triple | triple 单次 | triple 请求接受 |
|---|---:|---:|---:|---:|---:|---:|
| V7 A6375，V9 冻结基线 | 100% | 95% | 12% | 0% | 4.67% | 47.33% |
| V7 B6125，V9 冻结基线 | 100% | 48% | 0% | 0% | 0% | 33.33% |
| V9 A6425，原噪声 50 update | 100% | 100% | 22% | 13% | 50.33% | 93.33% |
| V9 B6175，原噪声 50 update | 75% | 96% | 0% | 0% | 0% | 33.33% |
| V9 A6475，低噪声 50 update | 100% | 100% | 9% | 0% | 5% | 45.67% |

B6175 单跳/三跳确认复测仍为 0%/0%；A6475 为 9%/0%，确认三跳请求接受
48.33%。以上评估样本的 failure/body-contact/invalid 均为 0，不能推断总体零风险。
A6475 单跳恢复超时 91%，三跳超时回合 99%。

| A 候选质量指标 | A6425 | A6475 |
|---|---:|---:|
| 单跳恢复超时 | 78% | 91% |
| 单跳高度 ≥3 cm 回合数 | 25/100 | 32/100 |
| 单跳最终腿误差 >0.10 rad | 14/100 | 54/100 |
| 单跳最终踝误差 >0.06 rad | 0/100 | 0/100 |
| 单跳最终脚姿态 >3° | 6/100 | 8/100 |
| 三跳最终漂移 P90 | 9.12 cm | 5.94 cm |

失败/拒绝更多时漂移较小不能解释为任务质量改善。最终姿态合格也不能替代
按时、连续 5 s 的完整恢复。跨版本成功定义不同，不能把 V7 旧协议三跳 44%
与本表 V9 严格指标直接比较。

## 6. 失败诊断与经验

### 6.1 idle 探索噪声截断训练数据

A6375 与 A6425 各 100 回合零更新对照：确定性 ready=100%，原噪声 ready=0%，
0.25 倍及 0.10 倍噪声 ready 均为 100%。A6425 原噪声最长连续合格仅 1.48 s，
不足要求的 2 s；所有对照无终止。

原噪声 A/B 块高度、跳跃完成、纯恢复完成积分全部为 0，说明训练跑通不等于
学到了技能。A6425 评估改善可以来自姿态/站立更新，不能据此声称跳跃学习。

低噪声 A6475 训练积分（每环境均值，24 s 累计）：高度 +0.819957、跳跃完成
+0.396484、纯恢复完成 +0.409180、拒绝 0、失败 -0.013672、恢复超时 -2.013672。
这确认数据阻断解除，但最终行为退化；正奖励非零不是验收证据。

### 6.2 真实落地与合成恢复不匹配

A6475 的 20 s CPU BAM 回放接受第 7 s 请求，跳高约 3.254 cm，无身体触地或
invalid，但恢复超时、完成 0。8–20 s 腿最大误差和脚 pitch 始终超限，终值约
0.11252 rad / 3.710°；躯干约 1.038°，踝误差约 0.03716 rad。完整视频已保存，
本轮只审阅落地与末段关键帧，未逐帧审阅全部视频。

用户 A6425 可视化回放持续约 79.38 s：7 s 接受，17/27 s 拒绝，完成 0，
恢复超时、busy=false、无身体触地/invalid。独立 35 s CPU 回放复现相同结果。

| 第 17 s 请求前指标 | A6425 实测 | 门槛 |
|---|---:|---:|
| 躯干倾斜 | 1.530° | ≤3° |
| 腿最大误差 | **0.103160 rad** | **≤0.10 rad** |
| 踝最大误差 | 0.037902 rad | ≤0.06 rad |
| 脚 pitch | 2.337° | ≤3° |
| ready_time / stable_time | 0 s / 0 s | ready 要 2 s、恢复要 5 s |

后续拒绝的直接原因是腿误差持续超限，而不是仍在 busy。差值很小却持续存在，
等待更久不会自动恢复。本轮没有修改验收阈值掩盖该失败。
合成 recovery=100% 不等于跳后恢复完成；后续课程应覆盖实际落地分布。

## 7. 工程验证与未完成项

- 全量 CPU 回归：2026-10-08 为 298 passed、1 skipped。
- 2026-10-11 提交前两次全量复测均为 297 passed、1 skipped、1 failed：
  `test_hf_jobs_flag.py` 的 mjlab 训练入口探针子进程 SIGSEGV。该组独立复测
  12 passed；开启 faulthandler 后一次运行在另一个入口探针得到 Warp 导入
  `add_builtin → constraint` 的 SystemError。根因未定位，不能报告当前全量通过。
- 本次涉及的全部 Jump 回归在先注册 `mjlab_microduck.tasks` 后为 73 passed。
  直接独立运行 Jump 文件时，旧 cfg 测试依赖任务注册顺序，会报任务 ID 不存在。
- 三次 64 env × 5 update GPU smoke；61D/14D、50 Hz、实际 LR 与奖励符号通过。
- smoke-003 标准 ONNX 最大动作误差 2.3841858e-6；A6475 为 3.3378601e-6，
  均通过 1e-4 门槛，normalizer、形状、V9 严格元数据通过。
- 独立 Warp 缓存构建成功，未复现 V8 历史编译异常；没有验证全新依赖安装环境。
- 未执行真实落地反向课程、DR 课程、多 seed 最终验收、60 s 五跳交付验收或实机测试。

## 8. 保留模型、证据与复现命令

A6425 是诊断候选，不是最终通过模型：

- checkpoint：`artifacts/jump-v9/A-pilot-0050/checkpoint_6425.pt`
- SHA256：`06729ff26302064b2f58e81defb05b2709f9731b63d5b92741a3ae0e2611297c`
- ONNX：`artifacts/jump-v9/A-pilot-0050/policy.onnx`
- 已回滚 A6475 SHA256：`0a000b7b2fa3cafc0591254c465c3e5511cc8a969c392f61dd89f39fecb9c5d3`

本轮代码与文档提交到 Git；模型、日志、视频等仍在本地产物目录，受 `.gitignore`
排除，本报告不代表这些大文件已随代码上传。下列命令要求本地对应权重存在。

可视化自动三请求（不要额外按 J；在启动终端按 Q 退出）：

```bash
uv run scripts/infer_policy.py \
  --jump artifacts/jump-v9/A-pilot-0050/policy.onnx \
  --jump-at 7 17 27 --json-out /tmp/v9-A6425-viewer.json
```

CPU BAM 无窗口完整回放：

```bash
MUJOCO_GL=egl uv run scripts/infer_policy.py \
  --jump artifacts/jump-v9/A-pilot-0050/policy.onnx \
  --headless --seconds 40 --jump-at 7 14 21 \
  --json-out /tmp/v9-A6425-triple.json --video-out /tmp/v9-A6425-triple.mp4
```

GPU 严格三跳评估（单跳改为 `--sequence single` 并使用新的输出名）：

```bash
uv run scripts/evaluate_jump_v9.py \
  --checkpoint artifacts/jump-v9/A-pilot-0050/checkpoint_6425.pt \
  --sequence triple --episodes 100 --seed 123 --device cuda:0 \
  --json-out /tmp/v9-A6425-triple-eval.json
```

主要原始证据：`artifacts/jump-v9/state.json`、`baselines/*.json`、
`evaluations/*-123.json`、各块的 `configuration.json` / `progress.json` / `result.json`、
`preflight/readiness-A6425-noise.json`、`preflight/low-noise-diagnostic-summary.json`、
`preflight/export-check-A6475.json`、`preflight/cpu-A6475-20s.{json,mp4}`。
用户回放原文件 `/tmp/v9-manual-check.jso`，独立复现 `/tmp/v9-A6425-repeat-check.json`；
临时文件不保证长期留存，关键数值已写入本报告。

## 9. 下一版方向

1. 保留 A6425，先采集真实跳跃落地状态，按失败的腿误差与脚姿态分组。
2. 建立从接近 ready 状态开始的反向恢复课程，先验证物理可行性，再扩大扰动。
3. 保持严格评估契约，混合站立和单跳样本，避免恢复训练损伤已有起跳能力。
4. 单独监测 3 cm 高度裕量、完成延迟与逐请求接受率；通过短训行为门槛再扩大预算。
5. 最后再进入连续三跳、DR、多 seed、CPU BAM 长回放和交付验收。
