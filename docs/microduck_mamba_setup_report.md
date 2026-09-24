# Microduck mamba 环境与 RL 闭环验证报告

验证日期：2026-09-05（Asia/Shanghai）
仓库提交：`29e887ecfbf5d37144759e5a9f8a176dfb83d547`
目标任务：`Mjlab-Velocity-Flat-MicroDuck`

## 结论

闭环验证通过。新建的独立 `microduck` mamba 环境完成锁定依赖安装；官方策略集已下载并固定到 Hub commit；walk、ground-contact、roller 三类 MJCF 均能编译且网格资源完整；完整 CPU 测试集为 **199 passed, 1 skipped**；64 环境、5 iteration 的 GPU PPO smoke test 正常结束；最后 checkpoint 经仓库规定的导出路径生成 ONNX，并通过 61→14、有限值和 CPU BAM MuJoCo rehearsal 检查。

这只是工程闭环 smoke test，不代表 5 iteration 已学会稳定行走。训练输出中的 episode length 和 reward 是短跑结果，不能替代正式训练或实机验收。

## 主机与环境

| 项目 | 结果 |
|---|---|
| OS / kernel | Linux 6.17.0-22-generic x86_64 |
| GPU | NVIDIA GeForce RTX 4090，24564 MiB |
| NVIDIA driver | 580.126.09 |
| Python | 3.12.14 |
| mamba 前缀 | `/home/xense/miniforge3/envs/microduck` |
| Torch | 2.9.1，`torch.version.cuda=12.8`，CUDA 可用，1 张 GPU |
| Warp | warp-lang 1.12.0；运行时报告 CUDA Toolkit 12.9、Driver 13.0 |
| MuJoCo / MuJoCo Warp | 3.10.0 / 3.8.1 |
| mjlab | 1.3.0 |
| BAM | better-actuator-models 1.0.1 |
| ONNX / ONNX Runtime | 1.22.0 / 1.24.4 |
| rsl_rl | rsl-rl-lib 5.0.1 |
| NumPy / SciPy | 2.4.1 / 1.18.0 |

没有安装 Conda CUDA Toolkit；GPU 运行依赖来自锁定的 Python 包和宿主机驱动。mamba 环境占用约 8.1 GB，验证结束时 `/home` 仍有约 16 GB 可用；未删除缓存或其他环境。

## 可复现命令

以下命令均在仓库根目录执行，依赖来源只有 `pyproject.toml` 和 `uv.lock`：

```bash
mamba create -y -n microduck python=3.12
UV_PROJECT_ENVIRONMENT=/home/xense/miniforge3/envs/microduck \
  uv sync --locked

UV_PROJECT_ENVIRONMENT=/home/xense/miniforge3/envs/microduck \
  uv run list-envs

UV_PROJECT_ENVIRONMENT=/home/xense/miniforge3/envs/microduck \
  uv run --with pytest pytest tests/
```

5 iteration smoke test 使用 TensorBoard、本地日志、单张 GPU，并将每轮 checkpoint 保存：

```bash
UV_PROJECT_ENVIRONMENT=/home/xense/miniforge3/envs/microduck \
  uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 --env.seed 42 \
  --agent.num-steps-per-env 24 --agent.max-iterations 5 \
  --agent.save-interval 1 \
  --agent.experiment-name velocity_smoke \
  --agent.run-name microduck_mamba_20260905 \
  --agent.logger tensorboard --agent.upload-model False \
  --enable-nan-guard True
```

从最后 checkpoint 走仓库唯一的安全导出路径：

```bash
mkdir -p artifacts/smoke
UV_PROJECT_ENVIRONMENT=/home/xense/miniforge3/envs/microduck \
  uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file \
  logs/rsl_rl/velocity_smoke/2026-09-05_02-55-00_microduck_mamba_20260905/model_4.pt \
  --onnx-file artifacts/smoke/velocity_smoke.onnx \
  --num-envs 1 --device cuda:0
```

导出器自动把 actor 的 observation normalizer 烘焙进 ONNX；没有手工转换 checkpoint。

## 官方策略集

来源：[pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies/tree/main)。下载时 Hub 返回并记录的固定 revision 为：

```text
088524a64e2557dc453256b6071dbb9d23888802
```

本地目录：`artifacts/pretrained/microduck-policies/`。manifest 为 schema 2、`obs_len=61`、`action_len=14`，含 9 个策略；所有策略均通过仓库的 manifest 校验，并逐个通过 ONNX Runtime 的有限值、非恒定输出检查。

SHA-256（包含目录中的有效下载文件）：

| 文件 | SHA-256 |
|---|---|
| `.gitattributes` | 未作为策略校验对象记录 |
| `README.md` | `98b45ea81164d1e1a1dd82255207053b15cd6c69d922a1c5cf3387ce604d4b74` |
| `manifest.json` | `d0c36e7b71129dd617339c63bcb1d704eab282c8617ebf14c2013a01abfb2dda` |
| `alpha_ground_pick.onnx` | `ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872` |
| `alpha_sitstand.onnx` | `c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8` |
| `alpha_stand.onnx` | `1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b` |
| `alpha_walking.onnx` | `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c` |
| `ball_kick_left.onnx` | `d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a` |
| `ball_kick_right.onnx` | `147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af` |
| `roller.onnx` | `cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd` |
| `roller_crouch.onnx` | `a1a084be240469c76ac9d3fa44d4792f16d4b1da60398b3ecd3cfc5e2244d990` |
| `roulade.onnx` | `3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b` |

下载使用未认证 Hub 请求，出现了速率限制提示，但文件完整下载并通过以上校验；报告不包含任何 token 或账户凭据。

## MJCF / STL 模型验证

使用实际 MuJoCo `MjModel.from_xml_path` 编译：

| 场景 | robot XML | nq / nv | nu | joints / meshes | 结果 |
|---|---|---:|---:|---:|---|
| walk | `scene_walk.xml` → `robot_walk.xml` | 21 / 20 | 14 | 15 / 38 | 通过 |
| ground-contact | `scene.xml` → `robot_groundcontact.xml` | 21 / 20 | 14 | 15 / 38 | 通过 |
| roller | `scene_rollers.xml` → `robot_groundcontact_rollers.xml` | 25 / 24 | 14 | 19 / 37 | 通过，含 4 个 `passive_*` joints |

仓库内本次扫描到 43 个 STL，三个场景引用的网格均随 XML 成功解析，没有缺失资源。没有发现 `.urdf` 文件，也没有额外生成 URDF。该训练栈的机器人资源是 Onshape 导出的 MJCF：MJCF 直接描述 MuJoCo 的 body、geom、joint、sensor、actuator 和 contact；URDF 更适合通用机器人描述/导入，不能替代这里的 MuJoCo 专用接触与 BAM actuator 配置。因此本次验证继续使用仓库现有 MJCF。

## Smoke training 与导出结果

任务注册表列出了目标任务。训练实际使用 `device=cuda:0`、64 个环境、物理步长 0.005 s、环境步长 0.02 s，actor observation 为 61D、action 为 14D，BAM M6 actuator 为 14 个执行器。5 轮训练全部结束，训练器报告 iteration loop elapsed 约 3 秒（首次 Warp kernel 编译是额外启动开销）。

| iteration | mean reward | mean episode length | nan termination |
|---:|---:|---:|---:|
| 0 | 0.17 | 17.60 | 0.0000 |
| 1 | 0.12 | 31.65 | 0.0000 |
| 2 | 0.13 | 34.71 | 0.0000 |
| 3 | 0.15 | 36.17 | 0.0000 |
| 4 | 0.19 | 35.00 | 0.0000 |

训练日志中可见的 penalty reward 项在 5 轮中均为非正；没有 reward 计算失败、NaN 或异常退出。短流程的 reward 波动属于预期，不能解释为已获得可部署 gait。

产物：

- TensorBoard 与配置：`logs/rsl_rl/velocity_smoke/2026-09-05_02-55-00_microduck_mamba_20260905/`
- checkpoint：`model_0.pt` 至 `model_4.pt`；`model_4.pt` SHA-256：`e75652b7e873f643d8bc7e2dba622803bed8ae5893c911fc5a004abddd56b8d2`
- 导出 ONNX：`artifacts/smoke/velocity_smoke.onnx`
- 导出 ONNX SHA-256：`04b4a0bf9fd1244c92df73b89e5a1943e51dcc5275630717d44aee204dd5ce7c`

`velocity_smoke.onnx` 再次通过仓库 `check_onnx` 和 100 步 ONNX Runtime smoke gate，输入输出为 `61 → 14`，输出有限且随输入变化。随后复用了 `scripts/infer_policy.py` 中的 BAM M6 CPU 控制路径（固定 7.4 V、0.1 V/Nm voltage-drop gain），在实际 ground-contact MJCF 上运行 250 个控制步、5.0 秒：观测为 61D、动作为 14D，全程 `qpos/qvel` 有限；最终 trunk z 为 0.1141 m，最大绝对动作约 0.4316。该 rehearsal 是数值/接口闭环，不是对 5 iteration policy 的行走质量背书。

## 问题、处理与遗留风险

1. 受限执行沙箱的 loopback 初始化在普通命令执行时报告 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`。为完成需要访问 GPU、mamba 前缀和网络缓存的本地验证，相关命令使用了受控的外部权限；没有因此修改仓库代码。
2. `mamba run -n microduck pytest tests/` 报 `pytest: not found`，因为 pytest 不是项目运行时依赖。按仓库命令约定改用 `uv run --with pytest pytest tests/`，结果为 199 passed、1 skipped；没有把 pytest 写入项目依赖。
3. 训练与导出期间出现既有的 tyro actuator selector warning（14 个 joint 表达式也匹配 7 个 site）以及 Warp deprecation warning；任务、导出和测试均未因此失败。
4. Hugging Face 下载使用未认证请求并收到速率提示，但小型策略集下载完整、revision 和 SHA-256 已记录。需要在受限网络或高并发环境重现时，可使用有权限的 Hub 认证方式，但不要把凭据写入报告或仓库。
5. 工作区当前磁盘只剩约 16 GB；策略、日志和环境均保留，未主动清理。正式大规模训练前应先确认磁盘预算。

本次没有阻断问题。仓库代码和公共 API 未修改；新增的只有本报告，模型、日志和 checkpoint 均位于现有忽略目录中。
