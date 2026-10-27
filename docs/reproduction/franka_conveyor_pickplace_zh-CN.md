# Franka 传送带抓取放置任务：完整复现记录（中文）

> 文档状态：已完成任务的事实记录  
> 最后核对：2026-10-21  
> 仓库：`/data/CodePWC/pickplace`  
> 本文对象：Franka 从固定供料碗抓取物体，放入传送带上的移动目标碗，并返回 Home；以及其 MuJoCo 闭环 sim-to-sim 验证。

English version: [franka_conveyor_pickplace_en.md](franka_conveyor_pickplace_en.md)

## 1. 最终结论

本次复现已经走通两条闭环：

1. **Isaac Lab / PhysX 状态 Teacher**：在本机 RTX 3090 上以 4096 个并行环境、无相机、无 RTX 渲染的方式完成三阶段 PPO 课程训练。最终选中的 v3c 权重在 256 个确定性评估环境中成功 **248/256（96.875%）**，成功条件包含抓取、放入移动碗、释放、物体稳定以及 TCP 返回 Home。
2. **MuJoCo 闭环 sim-to-sim**：MuJoCo 每 20 ms 生成一次当前状态，控制器据此输出动作，MuJoCo 再执行动作；不是轨迹回放。Isaac 权重零样本迁移为 **0/20**，经过显式动力学/接触适配后，在验收种子上为 **20/20**，在另一组未参与适配的种子 2000–2019 上也为 **20/20**。

最终关键产物：

| 内容 | 路径 | 结果 |
|---|---|---:|
| Isaac v3c 最佳 Teacher | `artifacts/teachers/teacher_repro_v3c_4096_no_rtx/checkpoints/ppo_teacher_30015488.pt` | 248/256 |
| 最佳权重元数据 | `artifacts/teachers/teacher_repro_v3c_4096_no_rtx/checkpoints/ppo_teacher_30015488.json` | 含配置、哈希和评估 |
| MuJoCo 适配权重 | `artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt` | 混合闭环控制器 |
| MuJoCo 演示数据 | `artifacts/mujoco/adaptation/expert_demos.npz` | 64 回合、16,130 样本 |
| MuJoCo 正式验收 | `artifacts/mujoco/franka_v3c_adapted_20ep.json` | 20/20 |
| MuJoCo 留出种子验收 | `artifacts/mujoco/franka_v3c_adapted_heldout_seed2000_20ep.json` | 20/20 |
| MuJoCo 成功视频 | `artifacts/mujoco/franka_v3c_adapted_closed_loop.mp4` | 1280×720、50 FPS |
| 速度粗扫/细扫 | `artifacts/mujoco/franka_v3c_belt_speed_sweep.json`、`franka_v3c_belt_speed_sweep_fine.json` | 见第 10 节 |

这些产物都保存在当前仓库内，没有写入或修改 `/data/CodePWC/IsaacLab`。

## 2. 仓库、分支和下载问题

当前 Git 关系为：

```text
origin   https://github.com/husterPWC/pickplace.git
upstream https://github.com/BY571/pickplace.git
```

当前工作分支是 `main`。`origin` 目前只有 `main`，上游还存在 `upstream/action-space-ablation`。本次复现基于 `main`，**不需要为了复现再 fork 额外分支**。如果以后需要该实验分支，可直接从 `upstream/action-space-ablation` 建本地分支，不应把它混入当前已验收的 `main` 成果。

最初的错误：

```text
RPC 失败。curl 56 GnuTLS recv error (-9)
unexpected disconnect
fatal: 过早的文件结束符（EOF）
```

它表示 HTTPS/TLS 数据流在对象包传完前中断，通常是网络、代理或长连接问题，不是仓库损坏，也不是 Git 分支问题。安全处理顺序是：

1. 删除或改名仅包含失败残片的目标目录；不要删除已有正常工作区。
2. 先浅克隆：`git clone --depth 1 https://github.com/husterPWC/pickplace.git`。
3. 进入仓库后再 `git fetch --unshallow`，必要时分批取历史。
4. 保留 `origin` 指向自己的 fork，另加 `upstream` 指向原项目，后续开发推送到 `origin`。

当前仓库已完整可用；不要因为早期 clone 错误重新覆盖它。

## 3. 环境隔离和版本策略

用户已有 `/data/CodePWC/IsaacLab`，本任务明确不破坏它。实际采用仓库内隔离环境：

```text
/data/CodePWC/pickplace/.venv
/data/CodePWC/pickplace/third_party/IsaacLab
/data/CodePWC/pickplace/.cache
/data/CodePWC/pickplace/artifacts
/data/CodePWC/pickplace/weights
/data/CodePWC/pickplace/datasets
```

安装脚本锁定的核心版本为：

- Python 3.12；
- Isaac Sim 6.0.1；
- Isaac Lab `v3.0.0-beta2.patch1` 源码树；
- PyTorch 2.10.0 + cu128；
- TorchRL 0.14.x、TensorDict 0.14.x；
- MuJoCo 3.8.0。

激活方式：

```bash
source scripts/activate_local.sh
```

该脚本把 pip、uv、Hugging Face、运行缓存、权重和训练产物都定向到当前仓库。后续命令应继续使用这套环境变量，避免把大文件写入用户主目录。

本机驱动没有升级。实际规避方式不是降低任务质量，而是拆分能力：

- **PhysX 状态训练和无相机评估**在本机 headless 运行；
- **Isaac RTX 相机渲染**在本机禁用；
- 可视化使用 PhysX 轨迹的独立网格渲染，或 MuJoCo viewer/离屏视频；
- 真正的双相机大规模 rollout 留给高驱动的八卡服务器。

训练包装脚本中的关键开关是：

```text
env.num_envs=4096
worker.render=false
logger.backend=null
```

`worker.render=false` 只关闭视频渲染，不关闭独立的 256 环境确定性评估。训练日志中偶尔出现 `gpu.foundation.plugin` / `ICpuInfo` 初始化错误，但 headless PhysX 训练和无相机评估仍完成；不能把这一日志噪声误判为训练没有进行。

## 4. 原项目的完整方法与我们实际完成的范围

原项目方法是 Teacher–Dataset–Student 三阶段：

```text
仿真特权状态 Teacher（PPO）
          ↓ 相机环境中 rollout
图像 + 状态 + 动作 + 奖励 + done 数据集
          ↓ BC / IQL / TD3+BC
只使用可部署观测的视觉 Student
```

其中：

- Teacher 可以看到仿真中的物体位姿和抓取标志；
- Student 只应使用真机可获得的信息，例如双相机图像和机器人本体状态；
- Hugging Face 的 `Sebasdi/pick-and-place-conveyor` 是原 Franka 场景的相机离线数据，不是本次状态 Teacher 的必要输入。

**本次最终验收的 Isaac v3c Teacher 是在本机从零开始并逐阶段 warm-start 的在线 PPO，不是用 Hugging Face 离线数据替代训练。** 仓库中虽然有 Student/BC 产物和完整的离线训练代码，但“视觉 Student 达到验收标准”不属于这次 Franka 状态 Teacher + MuJoCo sim-to-sim 的最终结论。

这一区分对换机械臂非常重要：换成 JAKA 后，Franka 的图像数据不能代表新的外观、运动学、相机标定和动作分布，必须在 JAKA 仿真中重新 rollout 相机数据；但状态 Teacher 仍可先在无相机环境训练。

## 5. Isaac 任务定义

### 5.1 场景

- Franka 基座在工作台旁；
- 固定供料碗约位于 `(0.45, -0.10)`；
- 食物物体从供料碗中生成；
- 目标碗放在由棱柱关节驱动的托盘上，沿传送带 `+x` 方向运动；
- 标称带速 `0.08 m/s`，目标碗经过机械臂可达区；
- 物理步长 `0.01 s`，控制 decimation 为 2，即策略频率 **50 Hz（20 ms）**。

目标不是仅“碰到”或“把物体移到碗附近”，而是依次完成：

```text
接近 → 夹住 → 抬起 → 追踪移动碗 → 放入 → 张开 → 物体稳定 → TCP 返回 Home
```

### 5.2 观测：45 维特权状态

Actor 和 Critic 都使用 `proprio + belt + privileged`：

| 组 | 内容 | 维数 |
|---|---|---:|
| `proprio` | TCP 位置 3、TCP 四元数 4、夹爪位置 2、9 个关节相对位置、9 个关节相对速度、上一动作 7 | 34 |
| `belt` | 目标碗位置 | 3 |
| `privileged` | 物体位置 3、四元数 4、抓取标志 1 | 8 |
| 合计 |  | **45** |

传送带速度没有显式进入原策略观测，因为 Isaac 训练速度固定为 `0.08 m/s`。这也是速度变化时性能下降的重要原因之一。

### 5.3 动作：7 维末端增量

- 前 3 维：TCP 相对位置增量；
- 中间 3 维：轴角形式的 TCP 相对旋转增量；
- 最后 1 维：二值夹爪命令，正值张开、负值闭合；
- 输出经 `tanh` 限制到 `[-1, 1]`，末端 6 维再乘 `0.5` 的动作尺度；
- Isaac 内部使用差分逆运动学。

选择末端增量而不是关节动作，使策略接口较容易迁移到不同机械臂；但 IK 参数、TCP 定义、控制周期和低层跟踪性能仍必须重新匹配。

### 5.4 奖励与成功条件

最终 `simple_v3b` 是阶段式稠密奖励：

| 奖励项 | 权重 | 目的 |
|---|---:|---|
| `reach_food` | 1 | 接近物体 |
| `grasp` | 2 | 形成有效夹持 |
| `grasp_lift` | 5 | 夹住并抬高 |
| `transport` | 10 | 抬起后靠近移动目标碗 |
| `food_in_bowl` | 20 | 物体已释放且在碗内 |
| `return_home` | 10 | 放置后 TCP 回 Home |
| `success` | +150 一次 | 完整成功 |
| `bowl_disturbance` | -10 | 避免撞击目标碗 |
| `action_rate` | -0.01 | 平滑动作 |
| `joint_vel` | -0.001 | 抑制过快关节运动 |

最终成功条件为：

1. 物体位于目标碗几何范围内；
2. 夹爪已经释放；
3. 物体相对碗速度足够小并连续稳定 5 个控制步；
4. TCP 到 Home TCP 的距离不超过 `0.05 m`。

Home 使用 TCP 空间距离而不是七关节距离，因为 Franka 是冗余机械臂，末端 IK 不能唯一决定肘部构型。

失败包括：目标碗离开可达区、碗掉下传送带、碗倾倒超过 45°、物体掉下工作台，以及安全超时。

## 6. 为什么采用三阶段课程训练

公开仓库中的最终 v3c 配置引用了一个没有随源码或 Hugging Face 数据集发布的早期权重，不能直接从该权重开始。因此我们重建了完整 warm-start 链。

| 阶段 | 配置 | 初始化 | 主要变化 | 实际总训练量 | 选中权重 | 选中权重评估 |
|---|---|---|---|---:|---:|---:|
| 1：v2 | `config_repro_v2.yaml` | 从零 | 仅要求放入；机器人初始关节 ±0.02 rad | 599,982,080 | 510,001,152 | 249/256 = 97.266% |
| 2：v2r | `config_repro_v2r_long.yaml` | v2 的 5.10 亿步权重 | 初始关节扩大到 ±0.25 rad、速度 ±0.1 rad/s | 299,958,272 | 290,062,336 | 250/256 = 97.656% |
| 3：v3c | `config_repro_v3c.yaml` | v2r 的 2.90 亿步权重 | 加入放置后返回 Home，使用 `simple_v3b` | 299,958,272 | 30,015,488 | **248/256 = 96.875%** |

“训练跑到约 3 亿/6 亿步”和“选择了哪个最佳 checkpoint”是两件事。最终 v3c 最佳点出现在该阶段约 3000 万步，后面继续训练不等于性能一定继续提高，因此部署和 sim-to-sim 使用的是最佳权重而不是机械地使用 `final.pt`。

网络与 PPO 核心参数：

- Actor/Critic MLP：`45 → 512 → 256 → 128`；Actor 输出 7 维；
- 激活：ELU；Actor 分布为 `TanhNormal`，确定性评估取变换后的均值；
- 4096 并行环境，rollout 16 步；
- PPO 每批 4 个 epoch，mini-batch 32768；
- `gamma=0.99`，`GAE lambda=0.95`，clip 0.2，entropy 0.005；
- v2 学习率 `3e-4`，v2r/v3c 为 `1e-4`；
- 每约 1000 万帧保存 checkpoint，并由后台 worker 在 256 个环境上确定性评估。

本地启动命令：

```bash
scripts/run_teacher_curriculum_local.sh v2
scripts/run_teacher_curriculum_local.sh v2r \
  artifacts/teachers/teacher_repro_v2_4096_no_rtx/checkpoints/ppo_teacher_510001152.pt
scripts/run_teacher_curriculum_local.sh v3c \
  artifacts/teachers/teacher_repro_v2r_long_4096_no_rtx/checkpoints/ppo_teacher_290062336.pt
```

## 7. 3090 上并行训练与显存误判

训练确实使用了 **4096 个并行 PhysX 环境**。显存占用看起来不高并不代表没有并行，原因是：

- 状态 Teacher 没有 RGB 相机、RTX 光栅/光追缓冲和 CNN；
- 每个环境只保存几十维状态、动力学数据和短 rollout；
- 4096 环境的状态 PPO 与 4096 环境的双相机 PPO 是完全不同的显存量级；
- GPU 利用率和 frames/s 比单个时刻的显存占用更能判断并行训练是否工作。

不要为了“让显存看起来占满”盲目增加相机或环境数。首先依据吞吐、评估成功率和 PhysX buffer 是否溢出来决定规模。

## 8. 训练中踩过的核心坑

### 8.1 直接训练完整 v3c 容易失败

从随机策略同时学习抓取、跟踪移动目标、释放和返回 Home，探索链过长。早期完整任务训练没有得到预期效果，最终通过 v2 → v2r → v3c 课程拆分解决。

### 8.2 正则惩罚过大会让策略“学会不动”

运动惩罚曾经大约高 10 倍。随机策略尚未偶然发现抬起和放置时，`action_rate` 与 `joint_vel` 的负回报已主导总回报，策略熵从约 4.7 塌缩到约 -13.6，策略停止探索。最终权重降到 `-0.01` 和 `-0.001`。

### 8.3 transport 奖励必须以“已夹住且已抬起”为门控

早期策略会把物体卡在两指之间沿桌面推向目标，以此刷 `transport` 奖励，却不真正抬起。现在 `transport` 和 `transport_fine` 只有在 `grasped && food_z > lift_height` 时才生效。

### 8.4 只在放入时终止，学不到返回 Home

如果物体一进碗就结束 episode，策略从未经历“释放后继续运动”的状态，连续生产时会停在分布外姿态。解决方式是增加 `return_home` 奖励，并让 `success_requires_home=true`；episode 直到物体已稳定且 TCP 回 Home 才成功。

### 8.5 训练成功率、评估成功率和最终权重不是一回事

并行采样时的训练统计受自动 reset 和近期 episode 混合影响。正式选权重必须使用独立、确定性、多环境评估的 sidecar JSON。最佳 checkpoint 也可能早于最后一个 checkpoint。

### 8.6 early-stop 依赖后台评估 worker

`early_stop.success_rate` 读取的是 worker 的独立评估结果，不是训练统计；若 `worker.enabled=false`，自动早停不能正常判定。关闭本机 RTX 视频时应设置 `worker.render=false`，而不是关闭整个 worker。

### 8.7 停止训练要发送 SIGTERM

正常的 SIGTERM 路径会保存 final checkpoint、manifest 和最后评估。强制 KILL 可能留下缺少 sidecar 或 manifest 未收尾的运行目录。

### 8.8 相机数据的显存和磁盘成本非常高

两路 84×84 RGB 的原始 uint8 数据约为 42 KB/帧，一百万帧约 42 GB。Isaac 环境内相机观测若转为 float32，加上 root/next 和 rollout 缓冲，显存会进一步放大。相机 rollout 不应照搬状态 Teacher 的 4096 环境规模。

### 8.9 本机不升级驱动时不能强求 Isaac RTX 界面

训练使用无相机 headless PhysX；可视化通过两种替代路径完成：

- `scripts/render_franka_teacher_mesh_video.py`：读取 Isaac PhysX 记录的真实 link 位姿，再用本地 Franka 网格离线渲染，不依赖 Isaac RTX；
- MuJoCo viewer 和 `mujoco_sim/render_video.py`：提供实时窗口与 MP4。

网格渲染是对真实仿真位姿的再绘制，不是训练时的相机观测，也不应拿来训练视觉策略。

## 9. Isaac 到 MuJoCo：如何保证是真闭环

MuJoCo 不是播放 Isaac 动作序列，而是每一步执行：

```text
MuJoCo 当前状态
  → 组装完全相同的 45 维观测
  → 当前控制器输出 7 维动作
  → MuJoCo 差分 IK、夹爪和接触动力学执行 20 ms
  → 新状态
```

移植时对齐了：

- Franka 关节、限位、惯量和网格；
- TCP：`panda_hand + [0, 0, 0.1034]`；
- 45 维观测的精确展平顺序；
- 四元数顺序；
- 7 维动作语义和 `0.5` 尺度；
- 50 Hz 策略周期；
- 传送带、供料碗、移动目标碗和终止条件；
- Home TCP 与 5 cm 容差。

契约测试结果：固定 reset 时 MuJoCo TCP 与 Isaac 参考相差小于 **0.04 mm**；同一状态的第一步策略动作误差小于 **2e-6**。测试位于 `tests/unit/test_mujoco_sim.py`。

这些契约对齐后，零样本 v3c 仍然是 **0/20**：15 回合目标碗离开可达区，5 回合物体掉下桌面，而且没有成功抓取。这不是接口接错，而是执行器、IK、接触和夹爪动力学的跨引擎差异；恰恰说明验收测试能检测真实迁移失败。

## 10. MuJoCo 适配与验收

### 10.1 适配方法

1. 在 MuJoCo 中实现只使用当前 45 维特权状态的九阶段 waypoint 专家：`pregrasp`、`descend`、`close`、`lift`、`transport`、`place`、`release`、`retreat`、`home`。
2. 并行采集 64 个成功回合，共 16,130 个 `(状态, 阶段, 动作)` 样本。
3. 以 Isaac v3c Actor 权重 warm-start，将 9 维阶段 one-hot 拼到 45 维状态，训练 150 epoch 的阶段条件 Actor。
4. 按阶段逆频率采样，避免长阶段支配 BC；夹爪动作损失权重更高。
5. 运行时使用保守混合：`action = 0.95 × nominal + 0.05 × learned`。

必须准确理解最终控制器：它是 **95% 动力学辨识后的名义 waypoint 控制器 + 5% v3c warm-start 学习 Actor**，不是纯神经网络策略，也不是 Isaac 策略零样本成功。阶段状态由控制器内部根据当前闭环状态推进，不使用未来状态或回放轨迹。

### 10.2 grasp_assist 的边界

正式 MuJoCo 验收启用了 `grasp_assist=true`，用于稳定简化 MuJoCo 模型中的两指接触和约束切换。它只会在以下条件同时满足时连接物体：

- 物体已经距离 TCP 小于 3 cm；
- 两指已经物理闭合到夹持区间；
- 不是远距离“吸附”；
- 夹爪一旦张开立即解除。

释放到碗内时还会消除约束切换造成的微小横向冲量。这个机制适合验证闭环接口和 sim-to-real 流程，但若要声明“纯 MuJoCo 接触模型迁移成功”，必须另做 `grasp_assist=false` 的验收。

### 10.3 验收结果

| 测试 | 回合 | 成功 | 平均完成时间 |
|---|---:|---:|---:|
| Isaac 权重 MuJoCo 零样本 | 20 | 0/20 | — |
| 适配控制器，种子 0–19 | 20 | **20/20** | 4.917 s |
| 适配控制器，留出种子 2000–2019 | 20 | **20/20** | 5.033 s |

成功回合必须满足抓取、放置、释放、稳定以及回 Home，而不是只检查物体是否短暂经过碗上方。

### 10.4 传送带速度鲁棒性

每个速度测试 20 个随机回合：

| 速度（m/s） | 成功率 |
|---:|---:|
| 0.040 | 20/20 |
| 0.060 | 20/20 |
| 0.080（训练速度） | 20/20 |
| 0.085 | 19/20 |
| 0.090 | 16/20 |
| 0.095 | 14/20 |
| 0.100 | 13/20 |
| 0.120 | 4/20 |
| 0.140 | 1/20 |

当前经 20 回合验证的 100% 区间是 **0.04–0.08 m/s**。更高速时的主要失败不是完全不会抓取，而是目标碗过早离开可达区，控制器无法在 deadline 前完成放置和返回 Home。不能把单个成功视频外推成全速度范围鲁棒。

## 11. 可视化与复现实用命令

运行 MuJoCo 20 回合正式验收：

```bash
.venv/bin/python -m mujoco_sim.policy_runner \
  checkpoint=artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt \
  episodes=20 seed=0 randomize=true grasp_assist=true \
  out=artifacts/mujoco/franka_v3c_adapted_20ep.json
```

打开 MuJoCo 实时界面：

```bash
DISPLAY=:1 .venv/bin/python -m mujoco_sim.policy_runner \
  checkpoint=artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt \
  viewer=true realtime=true viewer_hold=10 episodes=1 seed=0 randomize=false
```

生成 H.264 视频：

```bash
DISPLAY=:1 MUJOCO_GL=glfw .venv/bin/python -m mujoco_sim.render_video \
  checkpoint=artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt \
  seed=0 randomize=false width=1280 height=720 fps=50 \
  out=artifacts/mujoco/franka_v3c_adapted_closed_loop.mp4
```

运行跨引擎契约单元测试：

```bash
.venv/bin/python -m pytest -q tests/unit/test_mujoco_sim.py
```

## 12. 对后续新任务和 JAKA 的直接经验

1. **建立独立任务，不覆写本任务。** 新配置、新 gym id、新奖励集、新 artifact 根目录；现有 v2/v2r/v3c 与 MuJoCo 验收产物保持只读基线。
2. **先做状态 Teacher，再做相机 Student。** 状态训练在当前 3090 上高效；双相机 rollout 到高驱动服务器。
3. **先验证环境和脚本专家，再烧 PPO 步数。** reset、输送方向、抓取几何、成功判定和 deadline 中一个错误，都可能表现为“算法学不会”。
4. **课程从容易到完整。** 静止目标/慢速目标 → 标称速度 → 速度与初始状态随机化 → 放置后 Home。
5. **移动抓取需要时间信息。** 新任务中方块在传送带上运动，视觉 Student 应使用 2–4 帧堆叠、显式带速/编码器，或时序模型；单帧无法可靠区分位置相同但速度不同的状态。
6. **Franka 权重只能 warm-start 可兼容部分。** 场景语义改变后观测和最优动作分布变化；换到 JAKA 后运动学和夹爪也变化，不能把旧成功率直接继承。
7. **相机数据必须按新 embodiment 重采。** 高驱动服务器上的 JAKA 双相机仿真应保存图像、机器人状态、动作、奖励向量、terminated/truncated 和 outcome；再把 shard 拷回 3090 训练 Student。
8. **MuJoCo 是中间关卡，不是 sim-to-real 终点。** 必须保留零样本失败、适配方法、是否启用抓取辅助和留出种子结果，不能只保留成功视频。

## 13. 产物保护规则

后续开发必须遵守：

- 不修改 `/data/CodePWC/IsaacLab`；
- 不覆盖 `artifacts/teachers/teacher_repro_*`；
- 不覆盖 `artifacts/mujoco/` 下的正式验收 JSON、权重和视频；
- 新任务使用独立配置、任务名和 `artifacts/moving_cube_static_bowl/`；
- 新权重、数据集和下载缓存继续留在 `/data/CodePWC/pickplace` 内；
- 当前 Git 工作区包含已有用户资产和实验改动，禁止用 `git reset --hard`、`git checkout -- .` 或重新 clone 覆盖；
- 每个 checkpoint 都应保留配置、步数、哈希、评估 sidecar 和来源权重，避免只留下一个无法追溯的 `.pt`。

## 14. 相关代码与证据入口

- 环境说明：`docs/environment.md`
- Teacher 流程：`pipeline/0_state_teacher/README.md`
- 本地课程启动器：`scripts/run_teacher_curriculum_local.sh`
- 三阶段配置：`pipeline/0_state_teacher/config_repro_v2.yaml`、`config_repro_v2r_long.yaml`、`config_repro_v3c.yaml`
- 最终奖励：`pickplace/reward_sets/simple_v3b.yaml`
- 成功/返回 Home：`pickplace/envs/mdp/terminations.py`、`rewards.py`
- Isaac 无 RTX 网格视频：`scripts/render_franka_teacher_mesh_video.py`
- MuJoCo 环境：`mujoco_sim/env.py`、`scene.xml`
- MuJoCo 专家与适配：`mujoco_sim/expert.py`、`adapt_train.py`
- MuJoCo 运行与视频：`mujoco_sim/policy_runner.py`、`render_video.py`
- MuJoCo 详细说明：`mujoco_sim/README.md`
- 契约测试：`tests/unit/test_mujoco_sim.py`
