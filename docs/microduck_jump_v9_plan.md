# Jump V9：历史复盘与训练计划

日期：2026-09-25。审计代码基线：`6b4d385`。

执行进展另见 [V9 执行记录](microduck_jump_v9_execution.md)。本文保留训练前计划，
实际测试、预算与达标情况以执行记录及原始 JSON 为准。

本文是训练前计划，不表示 V9 已实现、开训或通过验收。本轮读取了 V1–V8
报告、V3–V8 状态文件、V7/V8 原始评估 JSON、V8 训练日志、V7 源码快照及
当前控制器/训练/评估实现，并运行了纯 CPU 状态机和模型前向运动学检查。
本轮未重放全部历史 checkpoint，也未重新观看历史视频；行为描述来自已保存
记录。下列建议参数是待验证的起点，不是已验证的最优参数。

## 1. 下一版目标

V9 固定 **3 cm 有效跳跃 → 双脚落地 → 完整恢复 → 可再次响应请求**。
训练高度 shaping 目标保留 3.2 cm，实际验收高度严格为 3.0 cm。
先修复状态机、测量、奖励和监督闭环，再分阶段训练恢复和连续三跳。
本版不进入 4–10 cm 升高课程。

现有 V8 检查点已退化；新训练从重新评估合格的 V7 候选开始。
当前首选候选是：

- A：`artifacts/jump-v7/v5_block_2000_ledger_2505/checkpoint_6375.pt`。
- B：`artifacts/jump-v7/v5_block_1750_ledger_2255/checkpoint_6125.pt`。

A 在历史协议下完整三跳较好；B 的单跳和恢复倾角较好。两者必须先接受同一个
修复后的 V9 评估，不能直接把旧成功率当作新基线。

## 2. 历史记录支持的结论

| 版本 | 保存的结果 | 对下一版的启示 |
|---|---|---|
| V1 | 6000 iteration 后成功奖励接近零；平均回合约 2.03 s，未达 4 s 上限 | 只增加探索时间不足以解决落地；应分解失败阶段。早期平滑税压制探索是合理假设，历史未做单因素验证 |
| V2 | 从站立教师迁移，两次 750 update，评估均无起跳 | 站立先验值得保留，但模仿站立不能替代动态技能的探索和任务奖励 |
| V3 | 旧名义单跳 100/100；CPU BAM 3/3，但最大倾斜 24.72° | 请求状态机、事件奖励、延后平滑税形成了有效的技能发现路径；其 3 cm 阶段实际接受 2.5 cm，不能算严格验收 |
| V4 | 姿态课程后单跳成功 65%，高度 P10 2.446 cm | 同时加严多项约束存在明显退化风险；尚不能将退化唯一归因于某一项惩罚 |
| V5 | 精确 3 cm 单跳 100%；最后漂移 P90 6.089 cm、航向 P90 12.292° | 高度已会，质量会随训练退化；早期较好模型不能被最终模型覆盖 |
| V6 | 单跳 100%；最终漂移 P90 1.746 cm，但恢复倾斜 P90 6.967° | 位移约束有效，但叠加质量项并未同时解决全部要求 |
| V7 A | 单跳 60%；三跳单次 81.33%、完整序列 44%；零记录安全失败，序列漂移 P90 9.323 cm | 应保留已有技能，专门训练恢复与累积漂移。最后模型的完整序列率退至 31% |
| V7 B | 单跳 70%；完整三跳 29%；单跳恢复倾角 P90 3.373°，三跳 4.897° | 可作为偏重落地姿态的对照初始化 |
| V8 | 1505 个已记账 update；最后 B 单跳/三跳成功均 0，failure/invalid 均 100%；下一块编译失败 | 同时存在实现错误、奖励失衡和基础设施故障；直接续训不能检验设计本身 |

跨版本的成功定义不同，以上百分比仅在各自旧评估协议内成立。
V7 训练回合 24 s、第三次请求约 21 s，第三跳不可能得到完整的 5 s 恢复窗口；
其评估使用 30 s，因此还存在训练/评估时域差异。

## 3. 本次新核实的开训阻断项

### P0-1：恢复样本无法完成，采样时序和类别也不一致

`jump_v8_reset` 将恢复样本标记为 accepted/took_off/landed/settle_hold，
但 `peak=0`；`JumpController.update` 的 complete 仍要求满足跳高门槛。
CPU 复现：双脚稳定、所有姿态/速度误差为零，持续 8.2 s 后
`stable_time=8.2`、`complete=False`、`recovery_timeout=True`。

另外，当前 mjlab `_reset_idx` 先执行 reset event，后重采样 command。
reset event 读取的 `recovery_only` 可能属于上一个回合；命令重采样又把
recovery-only 分配为一次请求，而非零请求。因此“35% 恢复样本”的标签与
物理初态不能仅靠配置比例确认。

正确做法：一次采样同一份 episode 类别，供物理 reset 和命令使用；明确区分
`recovery_complete` 与 `jump_complete`。恢复样本不伪造起跳高度、不支付跳高奖励，
纯恢复回合不安排跳跃请求。训练测试要覆盖初次 reset、部分 reset 和连续 reset。
初始化目标 yaw/xy 应对应实际初始姿态，避免随机 yaw 被无意要求回正到世界零方向。

### P0-2：右踝索引错误，最大腿误差实际包含头部

当前 `jump_v2_update` 用 `(4, 9)` 计算踝误差。实际 groundcontact 模型
CPU 编译确认：4 为左踝，9 为右 hip_yaw，13 才是右踝。
同时 `joint_max` 对全部 14 个舵机求最大值，不是约定的 10 个腿关节。

正确做法：通过关节名称在 servo 视图中解析双腿、左右踝；训练、评估、CPU BAM
使用同一份测量定义。plain/backlash 实际模型均要测试，不能仅把 9 改成 13。

本次还测量了 HOME 脚 site 的世界姿态：左右 roll 约 -5.002°/+5.002°，
pitch 约 0°；当前 5° roll deadband 下 HOME 脚成本约 `4.1e-7`。
这排除了“HOME 脚坐标整体旋转 90°”这一猜测，但没有证明动态脚姿态测量正确。
V9 应按实际脚底法向和 HOME 参考标定误差，并单列足跟/足尖离地或支撑几何检查；
仅有脚部接触布尔值不等于整个脚掌落平。

### P0-3：严格评估和 CPU 回放漏测，选模输入缺字段

`evaluate_v7_checkpoint` 的 V8 分支及 `jump_rehearsal.py` 仅传入平均关节误差
和水平速度，遗漏 joint_max、ankle_max、foot_pitch_max、foot_roll_max。
控制器对缺失输入补零，导致这些门槛被绕过。

V8 评估 JSON 还缺少 foot_pose、ankle_error、leg_home_max、ready/complete
latency、recovery_timeout_rate、busy_stuck_rate 等验收字段。
门槛函数因此无法通过；分支排序中的缺失脚姿态值变为 `1e9`，失去实际区分度。
从未进入恢复的摔倒样本还会记录 `recovery_tilt=0`，不能将它解释为姿态优秀。

正确做法：建立有版本的必填 schema；严格模式缺字段直接判“评估无效”，不进入
最佳模型候选。为未发生事件使用显式 unavailable/计数，而非理想零值。
从真实物理量统一生成训练、评估、CPU 回放的控制器输入。
终止时冻结各环境结果，避免后续 reset 的位置污染失败回合的 final drift。
分别记录落地瞬间、恢复过程最大值、最终保持窗口，禁止改变统计窗口掩盖失败。

### P0-4：脚姿态成本主导训练，触地事件成本则被 dt 缩小

V8 A250 最后一个日志窗口：

- `jump_settle_foot_pose = -33.6161`；
- `jump_height_progress = +0.0244`；
- `jump_complete = 0`；`jump_recovery_potential` 打印为 `0.0000`；
- mean reward 为 `-1131.25`。

同一日志口径下脚姿态负项约为高度正项的 1378 倍。smoke 中脚姿态负项已经
从约 -9.71 增到 -26.34，却仍进入正式训练。日志打印零不证明 potential 精确为零，
但足以说明必须记录未四舍五入的分阶段回报。

脚姿态和腿 HOME 成本是未限幅平方函数；例如脚误差 30°、scale 3°时成本为 100。
正确做法是先检查测量，再采用有界稳健成本，并统计每个回合/请求的实际积分回报。
在守住失败约束的同时，按阶段逐步增加姿态成本；不能只检查 penalty 符号。

另一个反向问题：`jump_touchdown_displacement_cost` 返回一次性成本但没有除以
`step_dt`，而通用事件 wrapper 除以 dt。50 Hz 时该项被 reward manager 乘 0.02，
相比“事件质量 -0.50”的直觉小了 50 倍。V9 统一事件/连续项单位，并用一个已知
触地轨迹检查累计回报；修正单位后重新标定权重，不能机械放大后直接长训。

### P0-5：拒绝事件可能在奖励读取前丢失

当前环境顺序为 reward compute → command compute。
命令阶段调用 `mark_readiness_failure`，但下一物理 step 的 controller update
先将 `readiness_failure_event` 清零，奖励读取时已看不到上一拍拒绝事件。
CPU 小实验复现事件由 1 变 0。

正确做法：将拒绝事件排队并消费恰好一次，或让状态更新/事件消费具有明确顺序。
测试必须覆盖实际 manager 顺序，确认拒绝 -4 的事件质量真正计入回报，而非只验证
`mark_readiness_failure()` 调用后瞬时字段等于 1。

### P0-6：记录了回滚/降学习率，实际执行未跟随

`finish_block` 更新顶层 `state['checkpoint']`，worker 却始终读
`state['branches'][branch]['checkpoint']`；分支选择后该字段没有继续推进。
实际 B750、B1000、B1250 的 `configuration.json` 都从 B500 的
`checkpoint_6625.pt` 启动。这些预算表示累计执行更新数，不表示模型连续训练进度。

监督状态写学习率 `5e-5`，worker 没有接收该参数；本次读取 B1000 checkpoint，
算法和 optimizer param group 的实际学习率均为 `1e-4`。
此外，选择分支会覆盖 global_best，更新 best/检测回退的顺序也需要独立测试。

正确做法：唯一活动 checkpoint 指针；父 checkpoint SHA 与阶段版本必须写进
每个 worker 结果。加载 optimizer 后再明确设置并断言实际 LR。
先与冻结的旧 best 比较，再决定晋升/回滚；安全不合格或评估无效的模型不可成为
可交付 best。保留“诊断候选”与“验收通过模型”两个状态。
`configure_v8_consolidation` 虽然存在，当前 worker 总调用默认 `configure_v8`；
V9 的采样分布也必须验证实际下发，而非只修改监督状态中的 consolidated 标志。

### P0-7：GPU 编译和长跑恢复链必须重新验证

B1250 日志显示错误在 MuJoCo Warp 的碰撞 kernel codegen 链：
`primitive_narrowphase → plane_convex_wrapper → make_frame → orthogonals`，
最后为 `Value after * must be an iterable, not function`。
当前只确认了报错位置，未复现并定位版本/缓存/类型解析根因。

正确做法：从锁定依赖的干净环境检查 CUDA、BAM、MuJoCo/Warp 组合；保留原始
错误与配置，定位后再决定改依赖还是缓存。CPU 单测通过不能替代 GPU smoke。
中断恢复必须使用新 attempt 目录，处理已创建目录但无 result.json 的情况。

## 4. 保留的正确方法

- 保持 61D actor、14D action、50 Hz、BAM、normalizer-baked ONNX、无 action EMA。
- 保留站立迁移和 exact-zero standing 样本；站立辅助损失限定 standing 子集，并监控
  对请求动作的影响。其权重是否抑制探索需要对照，不能凭 V2 失败直接下结论。
- 保留有效双脚离地、起跳超时、首次触地、完整恢复的事件定义；高度只奖励新增前沿。
- 保留 V8 的独立逐踝/腿姿态、脚底姿态和完成后保持约束，但先修复测量和单位。
- 落地点漂移单独收费；恢复期主要鼓励停稳，避免持续要求走回起点而破坏 ready。
- 使用真实落地状态做反向课程，覆盖已观察到的倾斜/单腿失配；初始化状态必须物理有效。
- 按行为门槛晋级，保留全局最佳；学习是否进步看起跳、恢复和完整序列，不看总 reward 单项。
- 镜像约束适合对称跳跃，但 V6 同时改变多项奖励，尚不能证明 mirror loss 单独有效。
  V9 先检查命令镜像和关节映射，再固定该设置，避免与主实验同时改变。

## 5. V9 的训练前工作与交付物

1. 修复上述 P0；新增实际模型关节测试、manager 时序测试、schema 缺项测试、
   worker 接续/LR/回滚测试。旧 V3–V8 控制器语义冻结，V9 显式版本化。
2. 在静态 HOME、闭环 standing、代表性真实落地姿态测量目标高度与可达姿态。
   每种至少保持 3 s，记录倾角、脚底支撑、关节误差和非脚部接触。
   如果 HOME/standing 无法满足拟定门槛，先定义物理可行的基准并重新冻结契约。
3. 新旧评估器都跑 A/B 候选：至少 100 个 standing、single、triple 回合，保存
   checkpoint hash、逐请求轨迹、失败分类和未剪辑代表视频。首次 V9 比较用同一模型、
   同一环境与 seed，分开记录评估器变化和策略变化。
4. 修复后的 GPU 64 环境 × 5 update smoke：动作/观测有限、奖励全项可算、符号正确、
   实际积分回报无异常支配、保存/恢复一致、ONNX 可导出并校验 61→14。
   再做短时 4096 环境构建/步进检查；5 update smoke 不足以证明技能学习。
5. 在 `artifacts/jump-v9/` 建立 manifest、源代码快照、运行配置、父模型 SHA、
   reward_mass、评估 schema 与预算台账。训练计数区分祖先累计、当前阶段和本轮实际消耗。

## 6. 分阶段训练与预算

以下为本轮建议上限：**4000 个新增 PPO update**，含 smoke、A/B 和回滚重试。
主训练按 4096 env、24 steps/update；一个 update 约 98,304 个环境转移。
不得因回滚重置预算，GPU 小规模诊断与评估时间单独记账。

| 阶段 | 更新上限 | 做什么 | 继续条件 |
|---|---:|---|---|
| GPU smoke | 5 | 64 env 验证完整训练/恢复/导出链 | P0 测试与 smoke 均通过 |
| A/B 小试 | 500，共 2×250 | 相同 V9 配置、同训练 seed，只比较两个父模型；每 50 update 评估 | 选安全且改善恢复的分支；两者均退化则停止改设计 |
| 恢复课程 | 750 | 真实落地 near-ready 初态开始，逐步扩大扰动；保持站立/单跳样本 | 恢复完成率 ≥90%，standing ≥99%，评估无摔倒/invalid，单跳不明显退化 |
| 完整单跳 | 750 | 串起精确高度、触地、5 s 稳定和再次 ready | 单跳完成率 ≥95%，请求接受 ≥99%，高度 P10 ≥3 cm，恢复/漂移质量改善 |
| 连续三跳 | 1000 | 先长请求间隔，再收紧到固定 7/14/21 s；回合给末跳留足恢复时间 | 中间三跳单次 ≥85%、完整序列 ≥70%，连续两个边界达标再巩固 |
| 随机化与巩固 | 750 | nominal 严格达标后，逐组引入训练基础随机化，混合 nominal 回归样本 | nominal 不退化，目标 DR 套件达到最终门槛 |
| 机动 | 245 | 重试、已证明有效阶段的短延长；也可不使用 | 总预算不能超过 4000 |

初始建议比例按“standing / recovery / single / triple”：

- A/B：30% / 10% / 60% / 0%。
- 恢复阶段：30% / 20% / 50% / 0%，确认学习后渐变至 20% / 35% / 45% / 0%。
- 单跳阶段：20% / 20% / 60% / 0%。
- 三跳阶段：20% / 15% / 35% / 30%；通过中间门槛后至 15% / 10% / 15% / 60%。

以上比例以实际 reset 类别计数校验。纯恢复的阶段晋级条件独立于跳高门槛；
完整单跳/三跳始终保留精确跳高验收。恢复可从更接近完成的真实状态开始，
而不是一开始把所有方向最大扰动同时施加。

连续训练初期建议请求 7/17/27 s、回合 42 s；能力形成后收紧至 7/14/21 s，
正式训练/评估统一用 40 s。独立抖动可能压缩相邻请求间隔，必须按实测
起跳、飞行、恢复延迟核验最小间隔。最终固定时刻的验收不得等待 ready 自动推迟。

建议从 `5e-5` LR 起步，加载后读取 optimizer 验证，暂用固定 schedule；先保持
继承策略的探索配置，记录动作 std，不同时大改探索、教师损失与镜像损失。
新奖励尺度会改变 critic 的回报分布，记录 value/advantage/KL；必要时对比重置
critic/optimizer 的独立小试，不能默认旧 optimizer 一定适合新任务。

## 7. 奖励与早停纪律

先保持事件主干：高度质量 +1.5、完整恢复完成 +4、失败 -8、起跳超时 -3、
恢复超时/请求拒绝 -4，作为 V8 设计的初始候选。所有事件只支付一次并验证 dt；
恢复辅助完成奖励独立核算，不重复计作跳跃完成。

恢复 shaping 保留有界势函数的有符号差分，改善正、恶化负、静止零；不能只截取
正差分。测试 reset、接触切换、终止与回合长度，排除循环刷分。
在 PPO 折扣下，普通差分不应直接宣称具有严格策略不变性。

脚姿态/HOME 使用有界成本，其权重从低值分阶段增加；建议新项从最终候选的
25% 起步，经行为评估到 50%/100%。action-rate 先保留较弱水平，技能质量稳定
后再增强。权重最终由每请求/每阶段积分回报确定，不照抄日志归一化值。

运行纪律：

- 每 25 update 存 checkpoint；前 250 update 每 50 update 做行为评估，之后每 100–250。
- 任何 NaN、缺失严格指标、关节映射错误、LR/父模型不一致、奖励符号错误立即停块。
- 评估安全失败率增加超过 5 个百分点，或完整单跳/序列率相对同协议基线下降超过
  10 个百分点，短复测确认后回滚。阈值是本版拟定操作规则，不是历史验证结论。
- 回滚恢复 actor、normalizer、critic/optimizer 等明确状态，随后应用计划 LR；
  worker 必须回报实际父模型 SHA 和 LR，确认后才能继续。
- 两个检查边界无恢复进展，先看具体失败：拒绝/不起跳/高度不足/单脚触地/
  足跟抬起/倾斜/漂移/超时。连续两次回滚仍复现退化，结束该配方，保留诊断视频。
- 比较模型先排除无效评估和安全不合格项，再比较完整任务成功与质量。无触地/无恢复
  的失败不能凭“漂移小、恢复倾角零”晋升 best。

## 8. 最终验收与交付

最终候选需连续两个检查边界通过，再在 seeds 123/124/125 各跑至少 100 个
single 和 100 个 triple 回合，另跑至少 100 个 standing 回合。
开发期间可用 123/124，125 留作最终检查。评估 seed 不等于独立训练 seed；
若要声称训练过程稳定，需要另预算训练 seed 43 的复现实验。

以下是拟冻结的 V9 严格门槛，P0 静态/动态可达性检查后再正式签入配置：

| 指标 | 目标 |
|---|---|
| 单跳完整成功率 | ≥95% |
| 三跳中的单次成功率 / 完整序列率 | ≥98% / ≥95% |
| 请求接受 | 所有预定请求接受；拒绝/未尝试均进入总分母 |
| 起跳高度 | 飞行期相对实测 standing 基准，P10 ≥0.030 m |
| 站立控制 | ≥99% standing 合格；无请求不主动跳 |
| 安全 | 此评估样本内零摔倒、身体触地、NaN/invalid；不是总体零风险保证 |
| 序列最终漂移 | P90 ≤1 cm，最大 ≤2 cm；并报告逐跳触地点漂移 |
| 最终航向变化 | P90 ≤3°，最大 ≤5°；另报全过程最大偏航 |
| 恢复倾斜 | 首次触地至下次请求/回合结束窗口，P90 ≤3°，最大 ≤5° |
| 脚底姿态误差 | 标定后 P90 ≤2°，最大 ≤4°；脚跟/脚尖接触几何另行审计 |
| 踝误差 / 腿关节最大误差 | 踝 P90 ≤0.05 rad、最大 ≤0.08；腿 P90 ≤0.10、最大 ≤0.15 rad |
| 静止水平速度 | 最终最大 ≤0.03 m/s |
| Ready / complete 延迟 | 从首次触地计，P90 ≤3 / 6 s，最大 ≤4 / 7 s |
| 超时与 busy | 起跳/恢复超时零；卡 busy 零 |

ready 要连续 2 s 满足：双脚有效接触、无身体触地、倾斜 ≤3°、腿平均误差
≤0.08 rad、任一腿关节 ≤0.10 rad、任一踝 ≤0.06 rad、脚姿态按标定定义
≤3°、水平速度 ≤0.03 m/s、yaw-rate ≤0.1 rad/s、roll/pitch 角速度范数 ≤0.5 rad/s。
严格验收对第一次及后续请求使用同一规则；训练早期若需宽松首次门槛，必须显式
标记 curriculum 状态，不得带入最终评估。complete 要完整保持 5 s，之后继续观察。

DR 按电压/摩擦、质量/CoM、编码器/IMU 噪声与延迟逐组加入，范围继承并审计
velocity 的真实训练配置；最终同一严格任务门槛用于声明支持的 nominal/DR 套件。
只通过 nominal 时交付名义仿真候选，不标为实机就绪。

通过后走仓库标准 normalizer-baked ONNX 导出；验证 61→14、数值有限、PyTorch/ONNX
动作误差（建议最大绝对误差 ≤1e-4）、元数据及三端状态机一致。
CPU BAM 至少 60 s、固定 7/17/27/37/47 s 五跳，另跑 7/14/21 s 紧间隔三跳；
全程录制，包括首次请求前和最后恢复后的保持，不手动挑选成功片段。
成功、失败或未完成均发布对应模型 hash、完整配置、评估 JSON、轨迹和结论。

## 9. 主要证据文件

- [历史汇总](microduck_jump_v1_v8_report.md)：V1–V6 的日志汇总与跨版本背景。
- `artifacts/jump-v7/evaluations/v5_block_2000_ledger_2505_123_{single,triple}.json`。
- `artifacts/jump-v7/evaluations/v5_block_1750_ledger_2255_123_{single,triple}.json`。
- `artifacts/jump-v7/evaluations/v5_block_2495_ledger_3000_123_triple.json`。
- `artifacts/jump-v7/source/source.tar.gz`：核对训练时的控制器/评估器定义。
- `artifacts/jump-v8/state.json`、`artifacts/jump-v8/*/configuration.json`。
- `artifacts/jump-v8/A_block_0250_ledger_0255.log`、`artifacts/jump-v8/smoke.log`。
- `artifacts/jump-v8/B_block_1000_ledger_1505/checkpoint_6875.pt`：实际 LR 读取。
- `artifacts/jump-v8/evaluations/B_block_1000_ledger_1505_{single,triple}.json`。
- `artifacts/jump-v8/B_block_1250_ledger_1755.log`：编译失败堆栈。
- [MDP](../src/mjlab_microduck/tasks/mdp.py)、[控制器](../src/mjlab_microduck/jump_control.py)、
  [评估器](../src/mjlab_microduck/jump_evaluation.py)、[课程](../src/mjlab_microduck/jump_curriculum.py)。
- [监督状态](../src/mjlab_microduck/jump_v8.py)、[监督 worker](../scripts/supervise_jump_v8.py)、
  [训练入口](../scripts/train_jump_transfer.py)、[CPU 回放](../scripts/jump_rehearsal.py)。

历史报告证据索引里的 V5–V8 `run/state.json` 与实际目录不同；本次读取的是各版本
根目录 `state.json`。本计划保留历史文件原样，避免将旧实验报告改写为新协议结论。
