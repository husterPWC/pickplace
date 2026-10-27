# JAKA ZU5 + AG95 静止小球抓取并放入 0.10 m/s 移动碗

> 最后更新：2026-10-26  
> 当前状态：状态几何主干已在 Isaac/PhysX 和 MuJoCo 中通过闭环验收；双 RGB-D 数据采集、视觉学生和真机闭环尚未执行。
> 两个 JAKA 十物体物流任务的统一过程、踩坑和 checkpoint 审计见
> [JAKA 双任务十物体连续处理复盘](jaka_zu5_ag95_ten_item_queue_zh-CN.md)。

## 1. 任务定义与冻结结果

本任务是原 Franka“静止物体、动态碗”流程的独立 JAKA 版本，不覆盖已经冻结的“动态小球、静止碗”成果：

```text
静止供料盘中的小球
  → 俯抓对齐
  → AG95 连续闭合并真实双指夹持
  → 净抬升
  → 跟踪 0.10 m/s 移动碗
  → 下降、释放、相对碗沉降
  → 竖直撤离
  → 返回 Home
```

正式约束：

- 移动碗速度固定为 `0.10 m/s`；
- actor 不输入碗速、球速、接触、力、触觉、电流或扭矩；
- 球起点在供料盘中心附近做 `±8 mm` 随机化；
- 不使用 grasp assist；
- 成功必须包含真实净抬升、球在移动碗内稳定、机械臂返回 Home；
- 夹爪开度和瞬时接触不能作为成功判据。

正式配置：

```text
pipeline/0_state_teacher/config_jaka_static_sphere_moving_bowl_m0_speed010.yaml
```

冻结权重：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/
  jaka_static_sphere_moving_bowl_m0_final_geometry_smoke16/
  checkpoints/ppo_teacher_final.pt
```

权重 SHA-256：

```text
d522d818dccdb7a073708b75ccb44fb7bc9b98a5780e10f057b5faba3b01555b
```

选择清单：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/selection.json
```

## 2. 最终验收

| 引擎 | 条件 | 严格结果 | 结论 |
|---|---|---:|---|
| Isaac/PhysX | 256 并行环境，seed 0，球起点 ±8 mm，碗 0.10 m/s | 250/256（97.656%） | 通过 |
| MuJoCo | 20 回合，球起点 ±8 mm，碗 0.10 m/s，CPU 闭环 | 20/20（100%） | 通过 |

Isaac 的 6 个失败均为 `bowl_exited_zone`；没有球掉桌、碗翻倒、碗离带或超时。当前控制器已超过验收目标，因此没有机械地延长 PPO。仅为生成可加载 actor 保存了 256 帧检查点；策略均值从零残差开始，主要能力来自已经验证的几何主干和确定性技能。

MuJoCo 20 回合报告：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/mujoco/
  strict_eval_speed010.json
```

最终视频：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/mujoco/
  speed010_strict_pick_place_home_full_scene_cpu.mp4
```

视频是 MuJoCo 逐步产生状态、控制器逐步输出动作、MuJoCo 执行动作的物理闭环，不是轨迹回放。`grasp_assist=false`，小球没有额外绿色描边。

### 2.1 10 个静止球分入 10 个动态碗的连续物流队列

在不覆盖单球任务的前提下，另建了连续队列 demo：10 个真实刚体球在 reset 时已经全部放在静止
低沿供料盘的 `2×5` 阵列中；10 个碗及其 10 个独立托板也从 reset 时起全部排列在加长传送带上，
以 `0.10 m/s` 同步运动。每轮只切换控制器关注的球—碗配对，不生成、瞬移、删除或重置物体。
已装球的碗继续向下游运动，并始终保留球—碗接触物理。

第一次按单球中心点周期估算使用 `0.63 m` 碗间距。静止供料阵列扩大了抓取位姿范围，实际周期增至
`6.9--7.44 s`；第四轮调用时目标碗已经前进至约 `x=0.25 m`，抓球后进入跟踪阶段过晚，结果为
`3/10` 后失败。该失败没有通过放宽判据或提前释放掩盖，而是按实测最长周期把正式间距改为
`0.75 m`。

正式 checkpoint 闭环使用选择清单中的 M0 权重，而不是零残差占位：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/
  jaka_static_sphere_moving_bowl_m0_final_geometry_smoke16/
  checkpoints/ppo_teacher_final.pt

SHA-256: d522d818dccdb7a073708b75ccb44fb7bc9b98a5780e10f057b5faba3b01555b
```

MuJoCo 每个 `20 ms` 控制步都从当前物理状态构造真实观测并调用该网络，再把非零网络残差送入
几何主干。10 轮合计调用网络 `3526` 次，`3526/3526` 步输出均非零；各轮平均绝对输出约
`0.00320--0.00390`，全程最大绝对输出约 `0.01357`。执行层严格使用 checkpoint 配置中的平移
残差上限 `0.03` 和旋转残差上限 `0.015`。

正式版本参数与结果：

```text
移动碗速度：             0.10 m/s
碗间距：                 0.75 m
传送带范围：             [-7.02, 8.13] m
传送带总长：             15.15 m
起始静止球：             10/10 全部存在
起始动态碗：             10/10 全部存在并运动
抓取—动态放碗—Home：    10/10
最终留在各自对应碗中：   10/10
中途生成或重置物体：     false
严格连续任务：           success
```

结果、视频和入口：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/mujoco/
  ten_static_spheres_ten_visible_moving_bowls_speed010_checkpoint_closed_loop.json
  ten_static_spheres_ten_visible_moving_bowls_speed010_checkpoint_closed_loop_full_scene_cpu.mp4
  ten_static_spheres_ten_visible_moving_bowls_speed010_checkpoint_closed_loop_full_scene_cpu.json

mujoco_sim/evaluate_jaka_ten_static_spheres_moving_bowls.py
mujoco_sim/render_jaka_ten_static_spheres_moving_bowls.py
```

视频先显示上游完整 10 碗队列，再显示供料盘中的 10 个静止球，随后切到工作区近景，最后用下游
广角确认 10 个装球碗。视频为 4 倍速 CPU Mesa 渲染，没有调用 GPU，也没有添加小球描边。

为避免把控制基线冒充网络策略，原零残差结果已明确另存为：

```text
ten_static_spheres_ten_visible_moving_bowls_speed010_zero_residual_baseline.json
ten_static_spheres_ten_visible_moving_bowls_speed010_zero_residual_baseline_full_scene_cpu.mp4
ten_static_spheres_ten_visible_moving_bowls_speed010_zero_residual_baseline_full_scene_cpu.json
```

正式结论只引用带 `checkpoint_closed_loop` 的文件。

## 3. 控制与观测合同

沿用已在 JAKA 动态小球任务中验证的主干：

```text
40 个规范 6D 抓取候选 × 每候选 20 个几何点
                    ↓
稳定目标筛选 + 几何接近控制 + 有界 PPO 残差
                    ↓
ALIGN → CLOSE → HOLD → LIFT
                    ↓
CLEAR → BOWL_HOVER → LOWER → RELEASE → RETREAT → HOME
```

关键点：

1. 小球是轴对称物体，抓取候选固定规范 yaw，避免候选姿态逐帧跳转。
2. 远距离接近、闭合、抬升、放置由显式阶段管理，PPO 只允许毫米级残差。
3. AG95 使用一个逻辑主关节，五个从动关节只在仿真执行层同步，不增加真机输入。
4. 碗位姿是当前状态输入；碗的速度不是输入。策略依靠连续位姿更新闭环跟踪。
5. 移动碗中的稳定判定使用球相对碗的速度，而不是球的世界绝对速度。碗以 0.10 m/s 运动时，碗内静止的球世界速度也约为 0.10 m/s。
6. Home 是显式尾段状态，不把“球已经入碗”提前当作完整成功。

本地 M0 不启用相机。正式真机部署前仍要在高驱动服务器建立双 RGB-D JAKA 场景，采集图像、状态教师动作、奖励和终止标志，再训练视觉学生。该步骤与当前 6D 抓取位姿不冲突：抓取位姿是状态教师/技能的动作目标，双相机负责在部署时从图像恢复策略所需的几何状态。

## 4. 最关键的失败与修正

### 4.1 MuJoCo 可达不等于 Isaac 低位可操作

最初供料盘放在：

```text
(x, y, z) = (0.30, 0.59, 0.024) m
```

这个位置在 MuJoCo 中能够 20/20，但 Isaac 64 环境严格评估为 0/64。单回合轨迹证明：

- 闭合前 TCP 与球心误差约 1 mm，说明抓取目标和接近奖励没有反向；
- AG95 确实闭合并接触小球；
- 抬升时 TCP 横向漂移，小球在供料盘中被拖动约 9 cm；
- 球只上升约 8 mm，未满足真实净抬升。

根因是该位置让 ZU5 在低腕高、近最大伸展处工作。MuJoCo 与 PhysX 的关节伺服、碰撞和低位奇异特性不同，不能用单引擎的“可达”替代跨引擎工作空间验证。

最终供料盘改为：

```text
(x, y, z) = (0.45, 0.00, 0.024) m
```

24 mm 垫高保留真实托盘支撑，同时使球心约为 5 cm，避开 ZU5 的低腕奇异区。仅改变供料几何、其余控制和严格判据不变，同一控制主干立即从 0/64 提升到 63/64，最终大样本为 250/256。

### 4.2 曾发现并拒绝的假阳性

早期探针曾让小球悬空在约 5 cm 高度，视觉上似乎容易抓取，但它没有真实供料盘支撑，不能代表真机任务。最终版本使用浅供料盘和实体 24 mm 垫高，小球由碰撞几何真实支撑。

### 4.3 不应因高成功率继续盲训

一次 PPO 更新的 KL 已经不小，而零残差几何主干本身达到 97.656%。因此当前阶段冻结短检查点，不启动 50 万步长训。以后若要优化剩余 6 个边界失败，应独立建立 M0a，小学习率、参考策略 KL 约束、短周期评估，并始终与当前权重做同 seed 对照。

## 5. 复现命令与硬件安全

本机 Isaac 只能使用物理 GPU1：

```bash
PICKPLACE_PHYSICAL_GPU=1 OMNI_KIT_ACCEPT_EULA=YES \
  scripts/run_jaka_static_sphere_moving_bowl_local.sh m0
```

日志中的 `cuda:0` 是隔离后的可见编号，实际映射到物理 GPU1。禁止省略 `PICKPLACE_PHYSICAL_GPU=1`，GPU0 有散热故障。

MuJoCo 评估显式屏蔽 GPU：

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  mujoco_sim/evaluate_jaka_static_sphere_moving_bowl.py \
  episodes=20 bowl_speed=0.10 randomize=true
```

MuJoCo 视频使用 CPU Mesa/Xvfb：

```bash
CUDA_VISIBLE_DEVICES=-1 LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw \
  xvfb-run -a .venv/bin/python \
  mujoco_sim/render_jaka_static_sphere_moving_bowl.py bowl_speed=0.10
```

上述流程没有升级驱动，没有修改 `/data/CodePWC/IsaacLab`，所有权重、报告和视频均保存在当前仓库中。

## 6. 高驱动服务器阶段

当前状态教师阶段已经完成。下一步把两个 JAKA 任务一起迁往高驱动服务器：

1. 动态小球 `0.06--0.08 m/s` → 静止碗；
2. 静止小球 → `0.10 m/s` 动态碗。

在服务器上为同一 ZU5+AG95 场景配置腕部与外部 RGB-D 相机，并行 rollout 冻结的状态教师，逐步保存双相机 RGB/深度、机械臂状态、教师动作、奖励和终止标志。之后回到本机 GPU1 训练或评估视觉学生。真机阶段预计需要两台相机：腕部相机负责近距离抓取细节，外部相机负责全局小球/传送带/碗定位。
