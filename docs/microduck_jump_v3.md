# Jump v3 执行说明

任务继续使用 `Mjlab-Jump-Flat-MicroDuck` 和同模型 Backlash 变体；BAM、50 Hz、61→14、无动作滤波。新增 critic 状态后维度为 91；旧 critic 不迁移。actor 从 `artifacts/jump_v2_transfer/standing_init.pt` 导入，冻结归一化器，critic、PPO 优化器从零开始。

站立基准由 `artifacts/jump-v3/preflight/standing_gate.json` 的100条存活闭环轨迹测量，保存在每个 checkpoint 中。不得用候选策略重新标定。所有新增成果在 `artifacts/jump-v3/`，不上传、不部署实机。

## 请求、奖励和评估

25% 纯站立，75% 单请求；请求在1–2秒调度，接受前连续稳定双脚支撑1秒。2秒仍未接受记就绪失败，接受后1.5秒未起跳记超时，两者保留在分母。忙碌期间忽略按键，不排队；成功落地稳定0.25秒再保持1秒才算完整完成。CPU和训练共用 `JumpController`。

双脚连续离地两步、两步均不超过45°且无非脚部触地，才算有效腾空。先检查有限值、姿态、接触，再更新峰值。小跳门槛严格为15 mm，20 mm阶段门槛20 mm，30 mm目标的最终门槛25 mm。

单请求实际累计正回报：支撑下向上速度新纪录≤0.1、有效起跳1、高度新纪录≤1、达标落稳2、完整完成1。首次跌倒/非脚部触地−2，随后终止；首次落地冲击−0.02×impact。事件/进展除以dt补偿奖励管理器，连续成本仍按时间积分。初期角速度−0.01、角动量−0.005、扭矩平方−0.001、动作变化0。平滑只能逐级0、−0.02、−0.05、−0.10，起跳率下降超过5个百分点回退。

评估预分配episode ID，提前结束后的重置回合永不占用原样本。每条记录与轨迹保留步后时间、请求、飞行、落地、落稳、跌倒/NaN、躯干/质心/脚底间隙、漂移、航向、冲击。checkpoint评估前后哈希必须一致。纯站立评估10秒：

```bash
uv run scripts/evaluate_jump.py --checkpoint-file <v3.pt> --standing --episodes 100 --seed 123 --json-out <standing.json>
```

## 训练与恢复

`train_jump_transfer.py` 是单块训练入口，一块最多250次PPO更新。`--init-checkpoint` 与 `--resume` 互斥；完整恢复保留优化器、教师、探索模式、随机数状态与更新计数。仿真回合会重新初始化，不是逐位连续。

```bash
uv run scripts/train_jump_transfer.py --output <新目录> --num-envs 64 --max-iterations 5
uv run scripts/train_jump_transfer.py --resume <v3.pt> --output <另一个新目录> --num-envs 64 --max-iterations 5
```

正式监督程序：

```bash
env -u PYTHONPATH MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  uv run scripts/supervise_jump.py --output artifacts/jump-v3 --hours 8
```

先检查预检验证文件；全局进程锁避免重复GPU实验。已有status时拒绝重新创建预算，不能用重启清零。监督程序原子写入 `status.json` 和 `budget.json`，记录PID、心跳、阶段、当前块进度、已保存与保守预留预算。每250次更新串行评估。正式训练的总预算4500次；烟测、探针、评估单独记账。

A/B分别从相同站立起点、相同seed=42训练250次：A请求时可学习std=0.15；B腿pitch/knee/ankle固定0.20，头颈0.03、其余0.05。非请求时二者头颈0.03、其余0.05。探索由采样观察的第48槽决定，PPO新旧log-prob、熵和KL一致使用该条件，不在采样后加噪。

按高度达标率、起跳率、有效峰值P90、低跌倒率选择分支，同分A。阶段0三次正常起点小跳达标率≥80%且站立≥95%；阶段1目标2cm一次通过后升3cm；阶段2三个完整成功率≥80%的checkpoint后恢复DR。DR强度25%、50%、100%，最后加入x/y ±0.08m/s推扰；nominal≥80%、DR≥70%才升级，100%需连续三个通过。早阶段节余可顺延，总量仍4500。

三个评估点无规定幅度改善时停止并保存诊断，不能通过延长无效配置或换成浅蹲回避任务。若无已验证的辅助初始状态，不自动从未经验证的探针状态重置训练。高度/DR与平滑升级不在同一边界。

所有被评估模型不可覆盖。课程升级另存 `*_continuation.pt`，同时记录evaluated_stage/next_stage。每块有独立配置快照与代码清单，不能覆盖之前快照。

## 交付

标准 `scripts/export.py` 导出；`verify_jump_export.py` 检查[1,61]→[1,14]及零输入/固定随机输入动作误差≤1e−4。最终候选需seed123/124/125每种nominal和完整DR各100回合、nominal站立每种seed各100回合达标。

CPU BAM完整30秒录像，前10秒站立，10.5、16.5、22.5秒请求。50fps，HUD与当帧步后物理状态同步，解码全帧并生成每秒全画面联系表。解码通过本身不等于动作视觉验收；未完成三次跳跃必须报告迁移失败。监督程序退出状态区分行为停滞、基础设施错误、预算耗尽和指标通过；最终由manifest与实际视频说明尚未完成的项目。
