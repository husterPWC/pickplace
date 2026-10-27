# 运动小球—静止碗任务（正式目标）

## 1. 任务目标

Franka 从传送带托盘上抓取直径 40 mm 的运动小球，将其放入机械臂侧面的静止碗中，松开夹爪，
等待小球稳定，并返回 Home。该任务自 2026-10-22 起取代运动方块，作为后续 JAKA + AG95
真机迁移的正式研究目标。历史方块代码、权重和实验记录保留为对照，不作为小球结果使用。

正式环境 ID：

- `FoodRobot-MovingSphereLift-v0`：截取、夹持和抬升课程；
- `FoodRobot-MovingSphereStaticBowl-v0`：完整抓取、投放、释放和回 Home 任务。

食物配置为 `moving_sphere`。它复用原任务已验证的 20 mm 半径球体和摩擦随机化，但不包含旧供料碗
的 reset 事件；运动小球的位置和初速度只由传送带任务重置。

## 2. 真机可部署观测约束

正式策略不输入传送带速度、物体速度、接触、力、触觉、电流或扭矩。传送带速度只作为仿真域随机化。
计划使用一个腕部 RGB-D 相机生成小球几何/点云与抓取候选，并结合：

- TCP 位置和姿态；
- 机器人关节位置；
- AG95 夹爪编码器；
- 上一步动作与短时位置历史；
- 固定标定的静止碗目标位置。

固定场景相机只作调试和冗余，不是策略运行的强制输入。

## 3. 已完成的物理验收

使用当前状态闭环专家，在 `0.10 m/s`、±5% 带速噪声以及物体/托盘/机器人随机化下测试 20 回合：

- 20/20 形成夹持并抬升至少约 3 cm；
- 19/20 到达约 0.203 m 的完整搬运高度；
- 15/20 完成入碗、释放、稳定和回 Home；
- 5 次失败分别发生在抬升、搬运和下降/释放阶段，表现为球体从平行夹爪中滑脱。

正式报告：`artifacts/moving_sphere_static_bowl/validation/expert_0.10_random_20ep.json`。
该结果证明运动小球任务物理可行，同时说明后续奖励和课程必须关注搬运阶段保持，而不能仅判断闭合。

## 4. 无速度几何闭环课程

所有阶段使用相同的 `grasp_point_gru` 网络输入和 7 维末端动作，确保严格 warm-start：

| 阶段 | 配置 | 目标 | 带速范围 |
|---|---|---|---|
| S0 | `config_moving_sphere_s0_lift.yaml` | 从头学习截取、真实夹持和 3 cm 抬升 | 0.045--0.075 m/s |
| S1 | `config_moving_sphere_s1_place.yaml` | 放入静止碗并释放，不要求 Home | 0.045--0.075 m/s |
| S2 | `config_moving_sphere_s2_speed.yaml` | 扩大初态和速度范围 | 0.06--0.10 m/s |
| S3 | `config_moving_sphere_s3_home.yaml` | 目标速度、完整释放和回 Home | 0.095--0.105 m/s |

S0 不使用方块 C0u 的“夹爪经过某段中间开度即成功”判据，而要求小球真实跟随 TCP 抬升 3 cm。
每阶段按独立确定性成功率选择最佳中间节点，不机械接受 final。

本机训练入口：

```bash
scripts/run_moving_sphere_curriculum_local.sh s0
scripts/run_moving_sphere_curriculum_local.sh s1 <s0-best.pt>
scripts/run_moving_sphere_curriculum_local.sh s2 <s1-best.pt>
scripts/run_moving_sphere_curriculum_local.sh s3 <s2-best.pt>
```

全部新权重与日志写入 `artifacts/moving_sphere_static_bowl/`，与方块成果隔离。

## 5. 实现验收（2026-10-22）

- 新注册环境的 4 环境 Isaac 测试通过：小球随 0.10 m/s 托盘前进，静止碗无漂移，半径和质量
  分别为 0.02 m、0.03 kg；
- 新任务 ID 下的正式 20 回合专家验收为 15/20，结果与切换前的对照完全一致；
- S0 运行 32 环境、1,024 帧 PPO 冒烟训练，完成两次采样和更新并保存 checkpoint；
- 冒烟 manifest 确认 actor/critic 不包含带速、物体速度、关节速度、接触、力或触觉输入；
- 冒烟产物位于
  `artifacts/moving_sphere_static_bowl/smoke/s0_32env_1024/checkpoints/ppo_teacher_final.pt`。

## 6. S0 首轮从零训练结果（2026-10-22）

正式 S0 使用 4,096 个并行环境训练 `29,949,952` 帧并正常结束，共保存 14 个周期节点和 final。
随后对全部 15 个 checkpoint 分别运行 seed 0、256 环境的确定性独立评估。所有节点的严格 3 cm
抬升成功率均为 `0/256`；绝大多数回合以运动物体离开抓取区结束。训练在线随机采样曾在约 800 万
和 1,200 万帧附近出现极少量成功，但确定性均值策略无法复现，因此不存在可接受的 S0 权重，不能
从 final 启动 S1。

这与方块 C0u 报告的约 90.6% 不矛盾：C0u 的成功条件只是 TCP 接近、夹爪开度进入中间区间并连续
保持 5 步，不要求物体离开带面。后续加入夹爪编码器稳定条件和严格微抬升后，C0u 触发率降到约
3%--9%，方块严格 5 mm 桥接阶段最佳仅 `2/256 = 0.78125%`。因此旧 C0u 的高数值是几何闭合
假阳性，不是真实抓取成功率。

必须区分：原项目“小球静止、碗运动”的旧 RL Teacher 曾达到 `248/256` 完整成功；运动方块和
运动小球的脚本专家也分别证明物理任务可行。当前尚未走通的是“无速度几何策略从随机初始化直接
探索运动物体真实抓取”的训练路径。下一版必须增加具有真实物体耦合验收的短距离桥接课程，不能
重复使用仅凭夹爪开度的成功判据，也不能继续盲目延长本轮 S0。

评估日志：
`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_s0_lift_4096_no_rtx/eval_all_seed0.log`。

## 7. sphere_gap S0a SAC 首轮失败审计（2026-10-23）

独立的 `sphere_gap_s0a` 路线实现了 40 个候选、每个 20 点的 6D 抓取表示、连续夹爪、隐藏双指
接触奖励、严格真实抬升 3 cm 成功判据和几何触发的闭合—撤回选项。64 环境冒烟训练能够完成
replay、反向传播和 checkpoint 保存；正式训练随后以约 50.3 万帧/小时运行。该轮在最后记录的
2,850,048 帧人工停止，所有可恢复 checkpoint（50--250 万帧）均保留，不能作为下一阶段 warm-start。

停止不是因为显卡、驱动或 Isaac 异常，而是出现了确定的学习塌缩：在线窗口的严格成功率始终为
0%，`alpha` 从约 0.1 降至 50 万帧的 `1.93e-5`、150 万帧的 `7.55e-7`、250 万帧的
`3.11e-35`，并在约 285 万帧下溢为 0。200 万与 250 万 checkpoint 的 actor 参数逐元素完全
相同，说明策略已经停止学习；Q2 在这一区间也几乎只更新最后一个标量偏置。

对 50、150、250 万 checkpoint 各做 8 回合确定性闭环诊断，全部为 `0/8`，且闭合—撤回选项
触发数均为 0。到最近有效抓取位姿的最小平均平移误差仍分别为 0.502、0.422、0.277 m，远大于
0.018 m 触发门槛；平均单步奖励只有约 `-0.6e-6` 至 `-1.2e-6`。这证明策略从未进入抓取区，
并非到达后夹持失败。

根因按优先级排列：

1. 接近奖励 `1-tanh(distance/0.05)` 在 0.4--0.5 m 初始距离处数值饱和为近零，远场动作没有
   可辨别的学习信号。
2. 初始策略标准差约 0.025，配合 0.02 m 末端动作尺度，每步随机位移约 0.5 mm；无远场奖励时，
   随机游走无法跨越数十厘米到达抓取区。
3. replay buffer 从未出现成功、接触或抬升样本；每个环境步 16 次更新只会更快过拟合近零奖励的
   失败分布，而不会自行创造正样本。
4. 自动温度没有下限和健康保护；`alpha` 下溢后熵项消失，近常数 critic 也使 actor 动作梯度消失。
5. 当前 replay 保存的是 option 变换前的策略动作，而仿真实际执行的是 option 变换后的动作。
   本轮 option 从未触发，所以它不是本轮零成功的原因；但若开始触发，这会让 critic 学到错误的
   `(state, action, next_state)` 对，必须在下一轮前修正。
6. 缺少自动训练健康门：连续若干评估选项触发为零、`alpha` 越界、actor 参数不再变化时，本应在
   10--20 万帧内停止，而不是运行到数百万帧。

下一轮长训前必须依次通过：远场奖励梯度测试；脚本成功轨迹写入 replay 后的短程过拟合测试；
option 执行动作与 replay 动作一致性测试；固定或有界温度测试；1--4 环境小样本达到严格成功；
最后才允许恢复 64 环境长训。传送带速度仍只作环境随机化，actor 仍不输入速度、接触、力、触觉、
电流或扭矩。

跨方块、小球、蒸馏、PPO 和 SAC 的完整因果复盘及 G0--G8 强制门禁见
[传送带动态抓取 RL 总复盘与强制训练门禁](rl_training_postmortem_and_gates_zh-CN.md)。

## 8. A0 近抓取 PPO 门禁（2026-10-23）

SAC 审计后新增独立 A0，不复用失败权重：4 个环境、0.01 m/s 固定低速、实测 Franka 近抓取
关节姿态、窄球位姿随机化、多尺度 6D 接近奖励、连续夹爪、无固定 option。actor 输入不含速度、
接触、力或触觉，成功仍是双指接触下真实抬升 3 cm。相关配置、启动方式和当前门禁状态见总复盘
文档第 7 节。

## 9. 冻结的最终技术路线（2026-10-23）

从本节点起不再在 PPO、SAC、行为克隆和纯 option 之间频繁切换。最终系统以原项目已经验证的
`状态 Teacher -> 课程训练 -> 相机数据采集 -> 视觉策略/部署` 流程为主干，并逐项吸收 GAP-RL
真正适用于动态抓取的模块。该决定同时适用于当前 Franka 验证和后续 JAKA ZU5/AG95 迁移。

固定的任务分解如下：

1. 腕部 RGB-D 或仿真真值生成一组物理有效的 6D 抓取候选；两种来源必须输出相同的数据契约。
2. 使用跨帧历史稳定抓取目标，避免候选逐帧跳变；传送带/物体速度仍只作环境随机化，不作为
   actor 显式输入。
3. PPO 策略负责闭环拦截和最终抓取位姿修正；先保留已验证的原项目训练基础设施，不再因单次
   失败直接更换强化学习算法。
4. 进入严格几何条件后，由明确的“最终接近、闭合、稳定保持、抬升”技能完成接触阶段。该技能
   必须具有可观测的进入/退出状态；若作为 option，训练转移必须符合 SMDP 定义，不能在策略动作
   之后偷偷改写执行动作。
5. 抓稳后继续使用原项目的分阶段课程完成运输、放入静止碗、释放和返回 Home。

允许仿真双指接触作为隐藏奖励、阶段标签和严格验收依据，但禁止输入 actor；真机默认最低配置为
一个腕部 RGB-D 相机、机器人关节/TCP 状态和夹爪位置编码器，不要求力、触觉、电流或外部物体测速。

A0 保持原配置运行，只承担“最近 6D 抓取位姿误差 + PPO 能否在极简分布中形成真实抓取抬升”的
诊断职责。A0 成功后逐项加入候选时间稳定性和明确闭合抬升技能；A0 失败则先用成功脚本轨迹校准
技能进入条件与阶段监督，再重新做短门禁。任何情况下都不以单纯延长训练或一次性更换多项变量
作为下一步。

### 9.1 A0 提前停止记录

A0 于约 195,328 帧（计划 500,000 帧的约 39%）人工停止。停止前在线完整回合始终为严格抓取
`0`、真实抬升 `0`，主要终止原因为小球离开抓取区；接近奖励由约 5 提高到约 7，但没有转化为
接触、夹持或抬升前缀事件。根据 G5/G8，继续训练只会优化接近代理指标，已经不能为“是否能学会
接触阶段”提供足够新增信息。

训练进程已正常退出，无残留 A0 进程。以下 checkpoints 与 manifest 保留用于失败诊断，不得作为
下一阶段 warm-start：`ppo_teacher_50176.pt`、`ppo_teacher_100096.pt`、
`ppo_teacher_150016.pt`。本次停止不改变第 9 节冻结路线；下一项工作应是用成功脚本轨迹校准并
验收明确的最终接近—闭合—保持—抬升技能，而不是延长 A0 或切换算法。

## 10. Stage G 稳定抓取门禁与根因闭环（2026-10-23）

新增 `pickplace/grasp_skills.py` 与 `scripts/validate_sphere_g0_skill.py`，控制器仅消费 40 个 6D
候选、TCP 状态和夹爪编码器；速度、接触、力、触觉、电流和扭矩均不是控制输入。仿真接触只用于
隐藏训练标签与严格验收。

球体绕竖直轴对称，40 个 yaw 候选在物理上大多等价。逐帧直接选择最近候选会造成编号切换和目标
朝向抖动，因此加入候选身份锁定和切换滞回。消融表明该问题真实存在但不是早期低成功率的主因：
候选锁定后随机速度门禁仍为 18/100。后续逐项定位得到：

| 版本 | 关键改动 | 0.01--0.03 m/s，100 回合 |
|---|---|---:|
| v1d | 可达的闭合力与按实际 TCP 高度结束抬升 | 18/100 |
| v1e/v1f | 远距离厘米级动作、近端毫米级动作 | 27/100、28/100 |
| v1g | 不再追球，改在 `x=0.45 m` 固定截获线等待 | 62/100 |
| v1h | 上游 8 cm 触发；无接触对齐增益 4.0、上限 8 mm | 100/100 |
| v1i | 再加入稳定保持和 8 mm 球心/TCP 耦合验收 | 100/100 |

v1i 覆盖实际速度 `0.01023--0.02972 m/s`；100 回合全部双指接触并真实抬升 3 cm，最大相对漂移
约 3.6 mm，最大指端力约 6.38 N。正式报告为
`artifacts/moving_sphere_static_bowl/validation/g0_skill_v1i_0.02_random_100ep.json`。

奖励和成功判据同步加严：先在带面附近稳定双指保持 5 步并锁存球心相对 TCP 向量，随后只有在
相对漂移不超过 8 mm 时，小球自身的高度进度才产生抬升奖励；接触丢失或滑脱立即停止奖励。严格
成功为同一耦合条件下真实抬升 3 cm，夹爪开度只用于执行器控制，不参与抓取或成功判断。

## 11. G0-S0 PPO 正式路线

新配置 `pipeline/0_state_teacher/config_moving_sphere_g0_s0_ppo.yaml` 保留原项目 PPO 主干。PPO
负责到固定截获区等待并进入稳定 6D 抓取盆地；达到 18 mm/0.15 rad 且连续稳定后，部署一致的
ALIGN--CLOSE--HOLD--LIFT option 接管接触敏感阶段。接近奖励也按当前位置分段：球在上游时奖励
固定截获悬停点，进入上游 8 cm 窗口后连续切换到最近 6D 抓取位姿奖励，全程不读取速度。

4 环境、1 次 PPO 迭代的真实 GPU 冒烟已完成，采样、状态化奖励、option、反向传播和 checkpoint
均正常。正式入口为：

```bash
scripts/run_moving_sphere_curriculum_local.sh g0s0
```

本阶段使用 1,024 并行环境、0.01--0.03 m/s、2,000 万帧上限，并每 200 万帧独立验收；按严格
成功率选择最佳节点，不机械使用 final。曾先做 4,096 环境容量测试，但 `4096 x 16 x 40 x 20`
点特征在整批 GAE critic 前向时额外申请约 13.28 GiB，3090 于首批 65,536 帧后 OOM；该检查点只
作诊断。1,024 环境实测 PyTorch 峰值约 14.24 GiB、吞吐约 1,578 万帧/小时，因此降并行数不改变
算法、课程或总训练帧数，只是适配 24 GiB 显存。

## 12. 技术来源、保留项与高驱动视觉阶段

当前方案是“原项目已经验证的训练主干 + GAP-RL 启发的抓取表示 + 本任务专用的稳定接触技能”，
不是对 GAP-RL 的完整复现，也不再用单次实验结果频繁替换算法。

### 12.1 参考 GAP-RL 的部分

1. 不再只用球心作为抓取目标，而是为物体生成物理有效的 6D 抓取候选，候选包含 TCP 位置、
   接近方向、夹爪闭合方向和宽度。
2. 将 40 个候选分别编码成 20 个确定性的各向异性高斯点，由点特征网络处理。这借鉴了 GAP-RL
   用高斯点表达一组 6D 抓取位姿的思想，使仿真真值和未来 RGB-D 抓取检测器可以共享输入契约。
3. 加入跨帧候选身份锁定和切换滞回，目标是获得时间一致的可抓取区域。对球体尤其重要，因为
   绕竖直轴旋转的多个候选物理等价，逐帧选最近候选会产生无意义的姿态跳变。
4. PPO 只输出末端残差动作和连续夹爪命令；进入严格几何盆地后执行短暂的对齐、闭合、保持、
   抬升技能。这保留 GAP-RL“抓取表示指导闭环策略”的核心思想，同时适配传送带的固定截获策略。
5. 仿真允许用双指接触作为隐藏奖励和验收标签，但不作为 actor 输入；部署侧由腕部 RGB-D 生成
   点云和抓取候选。这样训练时能判断真实抓持，真机又不依赖力、触觉、电流或物体速度传感器。

当前没有照搬 GAP-RL 的 SAC、完整 LocalGrasp、完整 Graspable Region Explorer 或其全部训练配置。
SAC 路线已经由失败诊断否决；当前仿真候选由球体解析几何生成，未来视觉阶段才替换为感知模块；
候选稳定目前采用适合球体的身份锁定/滞回，而不是声称已经复现完整区域探索器。

### 12.2 从已成功 Franka 任务保留的部分

1. 保留 Isaac Lab/PhysX 的大规模并行状态 Teacher 和 TorchRL PPO 主干。这套基础设施已经在原
   Franka 任务上通过 `v2 -> v2r -> v3c` 课程训练获得 248/256（96.875%）的确定性完整成功率。
2. 保留分阶段课程和 warm-start，但每一阶段只增加一个清晰能力：截获抓取抬升、运输到静止碗、
   放入释放、返回 Home，最后再扩大速度和随机化范围。
3. 保留“状态 Teacher -> 相机环境 rollout -> 离线数据/视觉 Student -> 部署”的总体流程。
   状态阶段先解决控制问题，视觉阶段再解决从 RGB-D 恢复可部署观察的问题，避免两类误差混在一起。
4. 保留末端笛卡尔增量控制、夹爪控制、并行仿真、周期 checkpoint、独立确定性验收以及按最佳节点
   选权重的规则；不假定最后一个 checkpoint 最好。
5. 保留完整任务的严格语义：必须是物体真实运动和真实抬升，之后还要放入、释放并返回 Home。
   夹爪开度不得作为成功代理，接触丢失或球相对 TCP 滑移后不得继续累计抓取/抬升奖励。
6. 保留成果和任务目录隔离；当前小球任务不会覆盖原 Franka 已成功权重、MuJoCo 验证结果或
   ZU5/AG95 资产和验证结果。

### 12.3 高驱动服务器上的相机 rollout 与视频

高驱动服务器阶段是正式流程的一部分，但必须在完整状态 Teacher 通过严格验收后再启动；当前
G0-S0 仍只处于“截获、抓稳、真实抬升 3 cm”阶段，不应提前用大量渲染掩盖控制问题。

服务器阶段按以下顺序执行：

1. 在相同 Franka、小球、传送带和静止碗场景中加载冻结的状态 Teacher，开启至少一个腕部 RGB-D
   相机；固定外部相机只作观测、调试和成片，可选而非策略必需输入。
2. 相机环境中做闭环并行 rollout，同步保存 RGB、深度、相机内外参、机器人关节/TCP、夹爪编码器、
   6D 抓取候选/高斯点表示、Teacher 动作、奖励、终止标志和回合结果。仿真接触可留作标签和诊断，
   但不得成为视觉 actor 输入。
3. 同时保存两类视频：腕部视角用于检查真实策略输入是否有效，外部视角用于检查截获、抓持、放置、
   释放和回 Home 的物理效果。数据集图像序列是训练资产，演示视频只是其可视化产物。
4. 使用这些数据训练视觉 Student（或执行必要的离线适配），然后仍在高驱动相机仿真中进行闭环
   验收并录制视频；通过后再进入 MuJoCo 闭环 sim-to-sim 和真机验证。

相机环境不需要照搬状态训练的 1,024/4,096 并行数；应按显存和渲染吞吐使用较小并行规模。
真机最低输入保持为一个腕部 RGB-D、机器人关节/TCP 状态和夹爪位置编码器，不要求第二相机、
显式传送带/小球速度、接触、力、触觉、电流或扭矩。迁移到 JAKA 时必须在 JAKA 相机仿真中重新
采集对应数据，Franka 的图像数据不能直接替代不同机械臂本体的数据。

### 12.4 本机 GPU 约束（2026-10-23）

本机物理 GPU0 因散热异常退出后续调度。`pickplace.app.launch_app()` 默认只暴露物理 GPU1，Isaac
Sim 和 PyTorch 进程内看到的逻辑设备仍为 `cuda:0`；日志或配置中出现 `cuda:0` 不表示使用了物理
GPU0。服务器或其他机器需要改变物理卡号时，必须显式设置 `PICKPLACE_PHYSICAL_GPU`。本机后续
训练、评估、渲染和相机 rollout 均不得将该变量设置为 0；代码检测到本机 `robot` 请求物理 GPU0
时会在加载 Isaac/CUDA 前直接拒绝启动。

2026-10-23 的 MuJoCo 视频验收中，OSMesa 因系统未安装软件 OpenGL 库而不可用；随后即使通过
`CUDA_VISIBLE_DEVICES=1` 将 EGL 显式限制到物理 GPU1，驱动监控也出现了无法通信。该 EGL 进程
已立即停止，未改驱动、未重启 Isaac、未继续尝试 GPU 渲染。此后本机 MuJoCo 物理闭环仍可在 CPU
运行；视频允许通过已核验为 `Mesa llvmpipe / Accelerated: no` 的临时 Xvfb 做 CPU 原生网格渲染，
或使用 CPU 状态渲染器，但禁止连接 NVIDIA EGL/GLX。在人工确认驱动完全恢复以前，禁止再次启动
任何 GPU 渲染任务。该条与“训练仅使用物理 GPU1”同等优先。

## 13. G0-S0 GPU 中断点与完整 checkpoint 验收（2026-10-23）

本轮原计划训练 2,000 万帧。物理 GPU0 散热异常前，最后一个完整保存的权重实际为
`ppo_teacher_14008320.pt`；随后进程在写退出权重时中断，目录中的 `ppo_teacher_final.pt.tmp` 是
不完整临时文件，禁止加载、重命名或作为 warm-start。所有完整 `.pt` 的 sidecar 与哈希均保留。

GPU 恢复后没有直接续训，而是在物理 GPU1 上对所有完整节点执行 256 环境、seed 0、确定性首回合
严格验收。结果如下：

| 帧数 | 严格成功 | `reach_food` | `grasp` | `grasp_lift` | 主要终止 |
|---:|---:|---:|---:|---:|---|
| 2,015,232 | 0/256 | 0.325 | 0 | 0 | 100% 离开抓取区 |
| 4,014,080 | 0/256 | 0.194 | 0 | 0 | 100% 离开抓取区 |
| 6,012,928 | 0/256 | 0.531 | 0 | 0 | 100% 离开抓取区 |
| 8,011,776 | 0/256 | 1.242 | 0 | 0 | 100% 离开抓取区 |
| 10,010,624 | 0/256 | 1.761 | 0 | 0 | 76.95% 离开、23.05% 掉落 |
| 12,009,472 | 0/256 | 0.847 | 0 | 0 | 100% 离开抓取区 |
| 14,008,320 | 0/256 | 3.504 | 0 | 0 | 100% 离开抓取区 |

网络学到了程度不单调但总体增强的接近行为，然而所有节点均未产生一次真实双指抓取或抬升，说明
策略始终没有稳定进入 `18 mm / 0.15 rad` 的 option 触发盆地。GPU 故障解释了为何训练在 1,400 万
帧附近停止，但不解释零抓取；继续从该节点机械训练到 2,000 万没有依据。下一步必须记录最小平移/
旋转误差、连续满足触发条件的步数和 option 状态转换，定位是等待目标、姿态盆地、触发滞回还是
动作尺度阻止了技能接管，然后只修改被证实的瓶颈并重新做短周期门禁。

## 14. 球体规范表示与粗细两阶段探索（2026-10-23）

针对球体绕竖直轴完全对称、40 个 yaw 候选物理等价但会造成目标编号/朝向切换的问题，新增
`FoodRobot-MovingSphereCanonicalGapLift-v0`。该任务仍保留网络所需的 `40 x 20` 高斯点张量契约，
但 40 个槽位重复同一个机器人友好的顶抓规范坐标系；奖励、`best_pose_delta`、全部候选误差和
ALIGN--CLOSE--HOLD--LIFT option 因而消费同一个目标。旧任务默认行为不变，历史 checkpoint 不会
被覆盖。相关几何与任务单元测试共 15 项通过。

规范表示的脚本闭环先通过物理门禁：固定 0.01 m/s 为 20/20，随机 0.01--0.03 m/s 为 100/100；
全部回合具有同时双指接触和小球真实抬升 3 cm，随机速度报告的平均最大球心/TCP 漂移约 3.29 mm。
这证明规范表示与既有稳定抓取技能在动力学上兼容。

随后从零运行 G0-S0c：固定 0.01 m/s、窄重置、规范表示、约 1 mm 初始笛卡尔探索，训练
1,998,848 帧。26 万帧与 final 各做 256 回合确定性严格评估，均为 0/256，闭合、双指抓取和
真实抬升均为 0。轨迹级诊断进一步证明 option 从未启动：

| 权重 | 最小平移误差（32 回合范围） | 最小旋转误差（范围） | 进入 18 mm/0.15 rad 门 | 最高阶段 |
|---|---:|---:|---:|---|
| G0-S0c 262,144 | 0.472--0.476 m | 0.792--0.801 rad | 0/32 | IDLE |
| G0-S0c 1,998,848 | 0.517--0.520 m | 0.741--0.748 rad | 0/32 | IDLE |
| 历史 G0-S0 14,008,320 | 0.082--0.298 m | 0.168--0.320 rad | 0/32 | IDLE |

因此 S0c 失败不能再归因于球体候选旋转；真正错误是把解决 7--18 mm 细门抖动所需的毫米级探索，
从一开始就用于约 0.5 m 的 Home 到截获区粗运动。历史高探索权重虽未达到 option 门，却已明显学会
粗接近。将该权重直接换到规范任务后，零样本误差仍为 0.089--0.303 m / 0.183--0.355 rad，说明
旧 actor 的粗能力与新表示兼容，可以用于 warm-start。

据此新增独立 G0-S0d 桥接阶段：从历史 14,008,320 帧权重 warm-start，在固定低速和规范表示下，
把探索重置为 `log(std)=-1.2`（约 6 mm 笛卡尔尺度），学习率降至 `1e-4`，每 25 万帧保存一次，
先跑 200 万帧短门禁。验收顺序固定为：先比较最小 6D 误差是否稳定缩小，再检查 option 是否触发，
最后才看双指耦合和真实 3 cm 抬升；若粗误差不再改善，不得机械延长。启动入口为：

```bash
scripts/run_moving_sphere_curriculum_local.sh g0s0d \
  artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g0_s0_ppo_1024_no_rtx/checkpoints/ppo_teacher_14008320.pt
```

G0-S0d 已在物理 GPU1 完成 1,998,848 帧。独立轨迹诊断表明 warm-start 后早期节点确实暂时保留了
更多粗接近能力，但没有进入严格门控，随后 PPO 再次遗忘并漂移：

| 权重 | 最小平移误差（16 回合范围） | 最小旋转误差（范围） | option 触发 |
|---|---:|---:|---:|
| G0-S0d 262,144 | 0.1266--0.1286 m | 0.381--0.392 rad | 0/16 |
| G0-S0d 1,015,808 | 0.301--0.303 m | 0.349--0.367 rad | 0/16 |
| G0-S0d final | 0.493--0.496 m | 0.691--0.705 rad | 0/16 |

因此 262,144 帧只可称为本轮“相对最佳粗接近节点”，不能称为抓取策略；final 更不能因为训练帧数
更多而被采用。下一阶段应使用已经通过 100/100 物理门禁的固定截获几何控制器作为粗运动基线，
让 PPO 只学习有界残差和进入细抓取盆地所需的修正，而不是再次要求纯 PPO 同时发现 0.5 m 粗运动
与毫米级抓取门。

### 14.1 G0-S0d MuJoCo 闭环 sim-to-sim

为观察跨引擎趋势，新增独立 `mujoco_sim/moving_sphere.py`，没有修改历史 v3c MuJoCo 任务。MuJoCo
每个 20 ms 控制步重新生成 TCP/关节/夹爪、规范 `40 x 20` 抓取点、6D 位姿差和 10 帧位置—动作
历史，PPO 根据该状态输出动作，再由同一 ALIGN--CLOSE--HOLD--LIFT option 和 MuJoCo 动力学执行。
这是真实闭环策略执行，不是脚本专家回放；actor 输入仍无显式速度、接触或力。

在固定 0.01 m/s、seed 0 的相同 45 秒条件下：

| 权重 | MuJoCo 最小平移误差 | 最小旋转误差 | option | 最大抬升 | 结果 |
|---|---:|---:|---:|---:|---|
| G0-S0d 262,144 | 0.484 m | 0.800 rad | 未触发 | 0 m | 超时 |
| G0-S0d final | 0.504 m | 0.785 rad | 未触发 | 0 m | 超时 |

早期节点在 MuJoCo 中略优于 final，但两者跨引擎后都失去 Isaac 中有限的粗接近能力，更没有闭合
或抬升。这既确认了后期漂移，也暴露出尚未做动力学适应时的 sim-to-sim 差距；当前视频只能用于
失败趋势诊断，不能作为成功验收。

旧“静止小球放入动态碗”项目的日志证明，成功视频使用同一个 `scene.xml`、
`mujoco_sim/render_video.py` 和 GLFW 原生网格渲染；其权重已经过专门的 MuJoCo 动力学适配并达到
20/20，而不是直接使用 Isaac 权重。新任务据此复用同一场景、STL 网格和相机逻辑，并通过临时
Xvfb 的 `Mesa llvmpipe` 做纯 CPU OpenGL 渲染。运行前 `glxinfo -B` 明确报告 vendor=`Mesa`、
renderer=`llvmpipe`、accelerated=`no`，脚本还强制隐藏全部 CUDA 设备。以下为正式完整场景视频：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0d_262144_mujoco_full_scene_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0d_final_mujoco_full_scene_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0d_early_vs_final_mujoco_full_scene_cpu.mp4`

早期生成的双视图 CPU 状态图仍作为数值诊断保留，但不再作为正式可视化交付：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0d_262144_closed_loop_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0d_final_closed_loop_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0d_early_vs_final_closed_loop_cpu.mp4`

每个单独视频旁都有同名 JSON，记录 checkpoint、速度、闭环属性、最小误差、option 状态和最终
结果。MuJoCo 新桥接与历史桥接合同测试共 7 项通过。

### 14.2 反向逃离的逐帧诊断与奖励缺陷

完整场景视频按 0、2、5、10、15、22 秒抽取六帧后，早期与 final 权重都显示 TCP 持续离开传送带
小球；正式拼图为：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0d_262144_motion_trend_6frames.png`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0d_final_motion_trend_6frames.png`

这不是相机视角造成的错觉，也不是 MuJoCo 把动作符号执行反了。262,144 帧权重的第一步平移动作
为 `[-0.916, -0.724, -0.844]`，而固定截获悬停目标相对 TCP 的方向为
`[-0.013, +0.300, -0.253]`；两者余弦只有 `0.016`，其中最关键的 y 轴方向相反。MuJoCo 实际 TCP
位移与策略动作的余弦为 `0.964`，随后多数步骤接近 `1.0`，证明控制器忠实执行了策略命令。
final 更差：第一步动作/目标余弦为 `-0.177`，4、8、15 秒附近分别约为 `-0.888`、`-0.910`、
`-0.939`，已经形成稳定的反向逃离策略。

也排除了四元数桥接错误。当前虚拟环境通过 editable install 实际加载仓库内
`third_party/IsaacLab` 6.1.14，而不是 `/data/CodePWC/IsaacLab` 的另一份旧源码；活动版本的
FrameTransformer 明确定义为 xyzw，与 `pickplace/envs/mdp/observations.py` 和当前 MuJoCo 桥接一致。

真正的奖励缺陷是：`sphere_intercept_pose_approach()` 返回非负的
`1 - tanh(平移误差/尺度 + 旋转误差/尺度)`。靠近时数值会增大，但远离只会渐近于 0，没有“比上一
步更远”的有符号惩罚。以旋转误差 0.8 rad 为例，平移误差从 0.02 m 增至 1.0 m 时，乘以
`reach_food=6` 后的瞬时质量从约 1.990 降到 0.0095，却始终不为负。同时：

1. `action_rate` 只惩罚相邻动作差，持续的饱和逃离动作一旦稳定就几乎不再受罚；
2. `joint_vel=-0.0001` 很弱；
3. 终止条件跟踪小球离开/掉落和严格抬升，没有 TCP 逃出工作区终止；
4. 远距离阶段同时耦合姿态误差，削弱了纯位置截获信号；
5. G0-S0d 没有参考策略 KL 约束，所以 warm-start 的有限粗接近能力会被 PPO 持续漂移覆盖。

因此不能再延长 G0-S0d，也不能使用 final。下一版在重新训练前必须先通过纯函数/脚本门禁：

1. 远距离只优化到固定截获悬停点的位置，进入近端窗口后才逐步加入姿态；
2. 加入有符号的势函数进度项，例如 `k * (d_prev - d_now)`：靠近为正、远离立即为负；
3. 对超过安全工作空间或连续远离设置显式惩罚/终止；
4. 增加很小的动作幅值正则，堵住“恒定饱和动作绕过 action_rate”的漏洞；
5. 按已冻结路线以 100/100 的几何截获器作为粗运动基线，PPO 只学习有界残差；
6. 单元测试必须比较相同状态下“朝目标一步”和“背离目标一步”的累计奖励，并保证背离严格更差。

在这些门禁通过以前，不启动新的长训练。

### 14.3 G0-S0e：几何基线 + 有界 PPO 残差（2026-10-23）

已建立独立任务 `FoodRobot-MovingSphereResidualGapLift-v0` 和配置
`pipeline/0_state_teacher/config_moving_sphere_g0_s0e_residual_baseline.yaml`，未覆盖 G0-S0d 或任何
旧权重。实现遵循固定路线：100/100 验证过的 `StableSphereGraspSkill` 全程负责
HOVER--ALIGN--CLOSE--HOLD--LIFT；PPO 仅在 HOVER/ALIGN 对机械臂动作添加有界残差，平移上限为
归一化 0.10（`action_scale=0.02` 时每步 2 mm），旋转上限为 0.05（每步约 1 mrad），策略永远不能
覆盖夹爪命令，闭合、保持和抬升阶段也不接受残差。

新奖励/安全合同如下：

1. 球进入截获提前窗口前，`reach_food` 只计算到固定 `x=0.45 m` 悬停点的平移距离，姿态误差完全
   不参与远端质量；进入近端窗口后才启用 6D 抓取位姿质量。
2. 奖励返回 `quality + k*(d_prev-d_now)/step_dt`，因此经 Isaac 的 `step_dt` 积分后精确增加
   `k*(d_prev-d_now)`；当前 `k=20`，靠近立即为正，远离立即为负，远/近目标切换步不做错误比较。
3. TCP 工作区为 x=[0.10,0.75]、y=[-0.20,0.48]、z=[0.02,0.75] m；越界立即触发
   `tcp_escape`。在同一目标阶段中单步远离超过 0.5 mm、连续 15 步也触发 `tcp_escape`。该事件映射
   到 `food_dropped=-100` 的一次性失败惩罚。
4. PPO 损失新增 `action_l1_coeff=0.002`，直接惩罚策略输出的原始残差绝对值，不会误罚几何基线；
   `action_rate` 仍保留，用于抑制抖动。

CPU 验收共 261 项全部通过；新增门禁证明相同起点下连续朝目标移动的累计回报为正、反方向为负，
且前者严格更高。另有远端姿态不变性、工作区各轴越界、连续远离计数、残差上界和夹爪所有权测试。
随后在物理 GPU1 上完成 4 环境、64 帧、1 次 PPO 更新的 Isaac 烟雾测试：环境、奖励、
`tcp_escape`、动作 L1 损失、采样、更新、checkpoint 和清理均正常，峰值显存约 0.135 GiB。

烟雾测试后，受限命令环境中的 `nvidia-smi` 一度报告无法与驱动通信且看不到 `/dev/nvidia*`；随后
在宿主机权限域复查确认这只是执行沙箱的设备隔离假象，并非驱动再次掉线。宿主机 GPU0 为 28°C、
空闲，GPU1 为 37°C，状态正常。因此正式短训允许继续，但唯一允许的本机入口仍是
`scripts/run_moving_sphere_curriculum_local.sh g0s0e`；脚本在主机 `robot` 上强制
`PICKPLACE_PHYSICAL_GPU=1`，进程内显示的 `cuda:0` 仅代表映射后的物理 GPU1。GPU 健康判断必须
使用宿主机权限域的只读 `nvidia-smi`，不能再把受限沙箱中缺失设备节点误判为宿主机驱动故障。

正式 G0-S0e 短训随后在物理 GPU1 完成 1,998,848 帧（1024 并行环境，约 6 分钟）。训练中首次
形成完整回合的 832 个环境全部严格成功，后续两个批次也分别为 100/100 和 92/92；所有已报告
批次的 `tcp_escape`、物体离带和超时均为 0。最终权重
`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g0_s0e_residual_baseline_1024_no_rtx/checkpoints/ppo_teacher_final.pt`
又通过 seed 0、256 环境、确定性首回合独立评估：256/256 成功，平均回合长度 1015.07 步，
`tcp_escape=0`、`cube_exited_zone=0`、`cube_off_belt=0`、`time_out=0`。该成功仍由双指稳定耦合与
小球真实抬升 3 cm 判定，不使用夹爪开度假阳性。评估结果已写入同名 checkpoint JSON。

### 14.4 G0-S0e MuJoCo 成功闭环视频（2026-10-24）

MuJoCo 桥接已按 G0-S0e 的实际执行合同更新：`StableSphereGraspSkill` 负责完整粗截获与
HOVER--ALIGN--CLOSE--HOLD--LIFT，checkpoint actor 只输出受限残差，夹爪命令仍完全归几何技能
所有。固定 0.01 m/s、seed 0 的闭环执行在 20.92 秒达到 `strict_lift_success`；球心相对回合初始
高度真实上升 0.0300717 m，验收门槛仍为 0.03 m，未使用夹爪开度代替抬升成功。

未经适配的 MuJoCo 执行曾稳定停在 0.0293832 m，比门槛低约 0.62 mm。原因是 MuJoCo 关节空间 IK
相对 Isaac 微分 IK 存在小的稳态跟踪偏差。桥接层因此把 TCP 执行抬升目标从 0.04 m 增加到
0.045 m，并延长执行窗口；checkpoint、actor 输入、球体实际抬升验收阈值和 Isaac 训练配置均未
改变。这是显式的跨引擎执行裕量，不应误写为 Isaac 权重零适配直接迁移。

正式视频及同名数值侧车为：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0e_final_mujoco_full_scene_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s0e_final_mujoco_full_scene_cpu.json`

视频为 1280x720、30 FPS、12.467 秒，按 2 倍速播放完整 20.92 秒物理回合并保留首尾定格。场景
包含 Franka 网格、夹爪、传送带、球和静止碗。录制前 `glxinfo -B` 确认为 Mesa llvmpipe、
`Accelerated: no`，且进程显式设置 `CUDA_VISIBLE_DEVICES=-1`，因此没有使用 GPU0 或 GPU1。
当前桥接仍启用 `grasp_assist=true`：满足 TCP 距离与夹爪开度条件后保持球/TCP 相对变换，以隔离
接触模型差异；所以该视频证明状态闭环、方向、时序和严格抬升任务走通，但不等价于无辅助的原生
MuJoCo 接触鲁棒性验收。G0-S0e 仅是截获--抓取--抬升阶段，视频不包含放入碗。

### 14.5 G0-S1：无速度输入的 0.01--0.03 m/s 课程（2026-10-24）

新增独立配置 `config_moving_sphere_g0_s1_speed_001_003.yaml`，从 G0-S0e final warm-start。相对
上一阶段只改变传送带域随机化：名义速度 0.02 m/s、噪声 0.5，即每回合均匀采样
0.01--0.03 m/s；actor/critic 仍没有显式速度、接触、力或触觉输入，球起点、机器人 reset、几何
基线、有界 PPO 残差和真实抬升 3 cm 判据均不变。`place_window=20 s` 与名义速度共同保持 0.40 m
空间抓取区。4 环境、64 帧烟雾测试先通过，正式训练只使用物理 GPU1；运行中实测 GPU0 为
28°C、0%、15 MiB，GPU1 为 67°C、99%、约 19.4 GiB。

训练首轮完成的全部回合从最快到最慢依次为 100% 严格成功，但约 129 万帧后新完成批次突然全部
离开抓取区。逐轮时序表明这不是策略突然漂移：失败恰好来自已经成功并由 Isaac 原生 auto-reset
进入第二回合的子环境。根因是 TorchRL 对原生自动重置调用 `_reset_on_native_autoreset()`，而六个
有状态控制边界 transform 只实现了普通 `_reset()`；因此 `StableSphereGraspSkill` 在第一回合结束
后仍停留于 `DONE`，第二回合不再输出截获运动。

已为所有有状态 action gate/option 补齐 native auto-reset 钩子，并保持按 `_reset` mask 只清理
结束的子环境。新增单元回归测试连同抓取技能/残差测试共 25 项通过。修复后使用最高速度
0.03 m/s、4 个环境、每个连续两回合验证，8/8 均严格成功；结果记录在
`autoreset_speed_0.030.json`。这条陷阱必须在迁移到 JAKA 或继续任何多回合训练时保留测试，不能
把生命周期状态泄漏误判为奖励、速度或 PPO 问题。

训练在诊断时停止，未机械跑满 200 万帧；冻结塌陷假象出现前最后一个 checkpoint：

`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g0_s1_speed_001_003_1024_no_rtx/checkpoints/ppo_teacher_1261568.pt`

修复 reset 后对该权重按相同 0.40 m 空间抓取区完成三个固定速度桶的 seed 0、256 环境确定性评估。
固定速度时 `place_window` 必须同步设为 `0.40/speed`，否则改变的是空间可达范围而不只是速度。

| 固定速度 | 严格成功 | 平均回合长度 | 离带/掉落/TCP 越界/超时 |
|---:|---:|---:|---:|
| 0.01 m/s | 256/256 | 1015.05 步 | 全部 0 |
| 0.02 m/s | 256/256 | 593.02 步 | 全部 0 |
| 0.03 m/s | 256/256 | 511.85 步 | 全部 0 |

三个桶合计 768/768，加上连续 auto-reset 门禁为 776/776。详细指标按 `speed_001`、`speed_002`、
`speed_003` 写入 checkpoint JSON 的 `eval_by_tag`，不会互相覆盖。G0-S1 因此验收完成，下一阶段
应从上述 1,261,568 帧权重 warm-start G0-S2，将速度域随机化扩到 0.03--0.05 m/s；仍不得加入
显式速度输入或同时扩大出生位置扰动。

### 14.6 G0-S2 直接扩速失败与 G0-S1b 重叠桥接（2026-10-24）

按原计划从 G0-S1 的 1,261,568 帧最佳权重 warm-start 独立 G0-S2，仅把传送带速度改为均匀随机
0.03--0.05 m/s。正式训练在物理 GPU1 上完成 1,998,848 帧并保留 8 个 checkpoint；GPU0 未参与。
独立固定速度验收发现，G0-S2 final 在 0.03 m/s 仍为 256/256 严格成功，但在 0.04 m/s 为 0/256，
全部触发 `tcp_escape`。G0-S1 原始权重在 0.04 和 0.05 m/s 同样均为 0/256；G0-S2 的 262,144、
507,904 帧节点在 0.05 m/s 也均为 0/256，失败轨迹和统计几乎相同。因此这不是单个 checkpoint
选择错误，而是课程入口形成了成功率断崖：连续均匀采样 0.03--0.05 时，恰好等于已掌握边界
0.03 的概率为零，训练几乎得不到严格抓取抬升正样本，200 万步也未能扩大边界。

该失败运行完整保留在：

`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g0_s2_speed_003_005_1024_no_rtx`

不从其 final 继续训练。新增桥接阶段 G0-S1b，重新从已验收的 G0-S1 权重开始，只把速度域改成
0.02--0.04 m/s；actor/critic 仍无显式速度、力、接触或触觉输入，其他变量保持不变。训练前用
seed 0、256 环境确定性探针测得该分布已有 177/256（69.14%）严格成功，失败项为 41/256
`tcp_escape` 与 38/256 超时。因此它同时保留足够真实正样本和足够困难样本，适合作为边界课程。
对应配置为 `config_moving_sphere_g0_s1b_speed_002_004.yaml`，启动入口为
`scripts/run_moving_sphere_curriculum_local.sh g0s1b <G0-S1-best.pt>`。只有 G0-S1b 在固定
0.02/0.03/0.04 m/s 三桶验收通过后，才重新进入 0.03--0.05 m/s；不得再用零重叠速度区间直接跳级。

G0-S1b 的 4 环境、64 帧烟雾测试和 39 项相关 CPU 回归测试通过后，正式训练在物理 GPU1 上完成
1,998,848 帧，8 个节点均已保留在
`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g0_s1b_speed_002_004_1024_no_rtx/checkpoints`。
训练内异步回合按速度/回合长度分批结束，不能直接用于 checkpoint 排名，仍需固定速度桶独立验收。

为直观看清原 G0-S2 在 0.05 m/s 的失败，使用 G0-S2 final 在 MuJoCo 中执行状态闭环并以 CPU Mesa
llvmpipe 录制完整网格场景。回合在 9.30 s 因 `sphere_exited_zone` 结束；TCP 到有效抓取位姿的最小
平移误差为 20.03 mm，技能已从 HOVER 进入 ALIGN，但始终未达到连续 4 步小于 7 mm 的闭合门槛，
夹爪没有附着，球的最大真实抬升为 0。视频与六帧趋势图为：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2_final_speed_005_failure_full_scene_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2_final_speed_005_failure_6frames.png`

这表明当前 0.05 m/s 失败不是早期版本的反方向逃离，也不是夹爪开度假成功，而是高速下跟踪误差
停留在约 2 cm，来不及进入 7 mm 稳定闭合盆地。MuJoCo 视频启用了与既有成功视频相同的
`grasp_assist`，但由于闭合条件从未满足，辅助从未触发；因此本次失败定位不依赖辅助抓取。

### 14.7 G0-S1c：提前下降并在截获点会合（2026-10-24）

0.05 m/s MuJoCo 逐帧视频显示，旧固定线截获器先在 `x=0.45 m` 的 8 cm 悬停点完全等待，随后才
下降；等 TCP 稳定后球已经接近或越过抓取窗口。该缺陷属于几何基线时序，继续让 PPO 从过晚基线
学习不会自然解决。旧 G0-S1b/G0-S2 的代码、checkpoint、日志和失败视频全部保留，不覆盖，并作为
未来 JAKA 迁移时必须重复的“旧截获器/新截获器”过渡 A/B。

新独立配置 `config_moving_sphere_g0_s1c_early_intercept_002_004.yaml` 保持 7 mm 连续 4 步闭合门槛、
双指稳定耦合、8 mm 相对漂移和真实 3 cm 抬升标准不变，只修正控制时序：

1. TCP 在球距离截获线 16 cm 时开始按球的位置进度下降，不再先完全停在悬停点；
2. 到提前 4 cm 的最终对准线时下降至 2 cm 预抓取净空，并与球在截获位置附近会合；
3. HOVER 到 ALIGN 的切换只检查粗位置，腕部姿态在下降中并行收敛；CLOSE 仍严格要求平移小于
   7 mm、旋转小于 0.15 rad 并连续保持 4 步，绝不提前乱闭合；
4. Franka 从经过测量的顶向下传送带待机位启动，避免通用 Home 的约 0.795 rad 初始腕部误差消耗
   高速截获时间。未来 JAKA 必须为自身运动学重新标定待机位，不能复制 Franka 关节角；
5. 闭合稳定后锁存球—TCP 的相对 XY 几何，抬升时保持该偏置，而不是错误地继续把球心追到 TCP
   中心；8 mm 漂移参考也在要求的 5 步稳定抓持结束时锁存，而不是使用第一次瞬态指尖接触位置。

actor/critic 仍无显式传送带速度、物体速度、接触、法向力、触觉、电流或扭矩输入；接触只作为
仿真训练和独立验收的隐藏标签。PPO 仍只能在 HOVER/ALIGN 输出有界残差，不能接管夹爪、稳定保持
或抬升。

诊断过程中保留了所有反例 JSON。重要结论包括：通用 Home 会使腕部来不及收敛；从第一次瞬态
接触计算漂移会错误消耗高速自定心位移；把闭合目标从 `-0.15` 加强到 `-0.20` 会横向挤飞球；把
抬升放慢到 120 步不能解决滑移；从带面开始限制 TCP 只领先球 2--7 mm 会把球压在带面，仅抬升
约 5--10 mm；几何 Z 反馈在严格漂移与 3 cm 抬升之间形成敏感折中，因此保留为默认关闭的研究
参数，不进入正式 G0-S1c。诊断脚本曾漏传新 XY 锁存开关，产生了一次与旧结果完全一致的无效运行；
现已补齐显式参数，提醒 JAKA 阶段必须检查“正式配置”和“诊断覆盖”是否一致。

正式方案在固定 0.04 m/s（G0-S1c 的 0.02--0.04 m/s 上界）完成两道 seed 0 严格门禁：

| 控制 | 严格成功 | 失败项 |
|---|---:|---:|
| 纯提前截获几何基线，PPO 残差为零 | 256/256 | 全部 0 |
| G0-S1 1,261,568 帧权重 + 原有 0.10/0.05 有界残差 | 256/256 | 全部 0 |

两组均达到双指稳定耦合与真实 3 cm 抬升，且 `tcp_escape`、离带、掉落和超时均为 0。对应报告为
`diagnostics/g0_s1c_formal_geometry_eval256_speed_004.json` 和
`diagnostics/g0_s1c_warmstart_residual_eval256_speed_004.json`。因此可从已验收 G0-S1 checkpoint
warm-start 正式 G0-S1c，无需重置 actor。固定 0.05 m/s 压力测试目前仍会在抬升耦合阶段失败，
它属于后续 G0-S2 边界，不得把 0.04 的桥接成功误写成 0.05 已解决。

所有 Isaac 运行只允许物理 GPU1；脚本在主机 `robot` 上强制 `PICKPLACE_PHYSICAL_GPU=1`，进程内
打印的 `cuda:0` 是重映射后的物理 GPU1。不得使用散热故障的物理 GPU0，也不得升级驱动或改动
现有 IsaacLab 安装。

G0-S1c 正式训练随后从上述 G0-S1 的 1,261,568 帧权重 warm-start，在物理 GPU1 上完成
1,998,848 帧（1024 个并行环境，约 5 分 53 秒），并保留全部 8 个 checkpoint。训练末段所有已
完成回合仍为 100% 严格成功，`cube_exited_zone`、`cube_off_belt`、`tcp_escape` 和 `time_out`
均为 0；峰值 CUDA 分配约 14.25 GiB。最终 checkpoint 为：

`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g0_s1c_early_intercept_002_004_1024_no_rtx/checkpoints/ppo_teacher_final.pt`

不能只按训练内异步批次选权重，因此又对 final 做了 seed 0、每桶 256 环境的确定性固定速度验收：

| 固定速度 | 严格成功 | 双指接触/稳定耦合 | 离带/掉落/TCP 越界/超时 |
|---:|---:|---:|---:|
| 0.02 m/s | 256/256 | 256/256 | 全部 0 |
| 0.03 m/s | 256/256 | 256/256 | 全部 0 |
| 0.04 m/s | 256/256 | 256/256 | 全部 0 |

三桶合计 768/768。报告分别为 `diagnostics/g0_s1c_final_eval256_speed_002.json`、
`diagnostics/g0_s1c_final_eval256_speed_003.json` 和
`diagnostics/g0_s1c_final_eval256_speed_004.json`。为排除“最后权重只是碰巧成功”，还在最难的
0.04 m/s 桶复评了 1,015,808 帧中期节点，同样为 256/256；final 的过滤后最小抓取误差中位数约
0.36 mm，中期约 0.46 mm，均远低于 7 mm 闭合门槛。因此冻结 final 作为 G0-S1c 正式权重，
中期节点保留作回退与 JAKA A/B 对照。

这里的 `true_lift_m` 诊断值约 0.028 m 是相对回合初始、已经略高于带面的球心位置计算；严格终止
条件仍按带面高度、球半径和 3 cm 净抬升计算，并且所有成功均经过稳定耦合计数，不能解读为放宽
了 3 cm 标准。当前结论只覆盖 0.02--0.04 m/s。随后对同一 final 补做固定 0.05 m/s、seed 0、
256 环境压力测试，严格成功为 0/256，全部超时。它不是截获或闭合失败：256/256 都进入
HOVER--ALIGN--CLOSE--HOLD--LIFT/DONE，均出现过双指接触和稳定耦合，小球相对初始位置的最大
真实抬升为 34.86--37.77 mm；失败原因是高速抬升过程中相对几何漂移，使稳定耦合与 3 cm 净抬升
条件未能同时维持。报告为 `diagnostics/g0_s1c_final_eval256_speed_005.json`。0.05 m/s 因此仍是
下一阶段 G0-S2 要解决的边界；不得降低 7 mm 闭合门槛或 3 cm 成功标准来制造通过。

### 14.8 G0-S2a/G0-S2b：抓后随动探针与长传送带提前入场（2026-10-24）

G0-S2a 没有覆盖 G0-S1c final，也未开始正式训练。针对 0.05 m/s 抓后相对滑移，先给几何技能加入
可选的目标位移前馈、抬升前馈锁存和负载后几何重锁存；actor 仍不接收显式物体/传送带速度、接触、
力或触觉。多组 64 环境短探针均为 0/64 严格成功；最新组合虽 64/64 曾稳定耦合并达到约
35.7--38.5 mm 相对初始位置抬升，仍因抬升期间相对漂移而超时。最新报告为
`diagnostics/g0_s2a_probe_latched_xyff1000_gain001_eval64_speed005.json`。因此不得宣称 G0-S2a
已解决，也不从这些探针盲目长训。

随后用同一 G0-S1c final 和最新可选随动参数做 MuJoCo 对照。原 `x=0.20 m` 起点在 0.05 m/s 下
最近平移误差已到 1.80 mm，但旋转误差最低仍为 0.202 rad，高于 0.15 rad 闭合门槛，9.3 s 时球
驶出抓取区。应用户提出的真实产线布局思路，独立建立 G0-S2b 诊断：保持截获线 `x=0.45 m`、速度、
策略权重和全部严格门槛不变，只把小球起点移到 `x=0.05 m`，并把 MuJoCo 传送带上游视觉边界从
`x=-0.10 m` 延至 `x=-0.25 m`。这使到截获线的理论准备时间从 5 s 增至 8 s。

seed 0 的闭环 MuJoCo 结果证明该变量有效：最小平移/旋转误差分别降至 0.189 mm/0.00059 rad，
成功进入 CLOSE/HOLD/LIFT/DONE 且抓取辅助触发，不再属于“追不上/来不及闭合”。但球的最大净
抬升只有 18.12 mm，未达到不变的 30 mm 严格标准，最终在 45 s 超时。因此当前结论是“延长上游
准备段解决了该 MuJoCo 回合的截获时序，但尚未解决抬升执行”，不能写成完整成功。视频和记录为：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2b_long_belt_start005_speed005_full_scene_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2b_long_belt_start005_speed005_full_scene_cpu.json`

该视频使用 CPU Mesa llvmpipe，未访问 GPU0。`grasp_assist` 与既有 MuJoCo 对照口径一致，故它只
用于观察截获与控制时序；是否真实跨引擎抓持仍须另做无辅助接触验收。下一步应先把相同提前入场
变量同步到独立 Isaac G0-S2b 配置，在物理 GPU1 做不训练的固定 0.05 m/s 多环境评估；只有 Isaac
严格成功率显示改善，才允许从 G0-S1c final 做短周期 warm-start。

逐帧复查上述第一版 G0-S2b 视频后发现两个实现问题，不能把 18.12 mm 停止误判成策略主动放弃：

1. 只后移了球和传送带视觉边界，漏掉了承载板；球从 `x=0.05 m` 出发，而板仍从 `x=0.20 m`
   出发。该错误由视频直接暴露，说明所有后续布局改动必须同时核对物体、载具、碰撞面和驱动目标；
2. 几何技能的 `lift_timeout_steps=300` 在约 6 s 后把未达到目标高度的 LIFT 强制切到 DONE，DONE
   又输出零位移动作，所以出现“球没有掉、机械臂却一直停住直到总回合超时”。

修正版仍保持球速 0.05 m/s、7 mm 闭合门槛和 30 mm 净抬升标准不变，将球与承载板统一置于
`x=0.05 m`，板的驱动位置命令也加相同初始偏移；`lift_timeout_steps` 放宽到 1200（24 s），总回合
上限放宽到 60 s。seed 0 的 MuJoCo 闭环在 10.1 s 即达到 `strict_lift_success`，没有耗尽任一新上限；
最大净抬升 30.019 mm，最小平移/旋转误差为 1.740 mm/0.0341 rad。修正版视频和记录为：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2b_pallet_synced_lifttimeout1200_speed005_full_scene_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2b_pallet_synced_lifttimeout1200_speed005_full_scene_cpu.json`

该单回合证明“提前入场 + 板球同步 + 足够技能预算”可解除本次 MuJoCo 时序失败，但因为仍启用
`grasp_assist`，不能替代多 seed、无辅助接触和 Isaac 严格验收。第一版错误结果及视频继续保留，
作为 JAKA 迁移阶段必须检查载具同步与内部技能超时的反例。

在不改策略权重和严格门槛的前提下，进一步做了 0.08 m/s 确定性压力测试。为了保持与 0.05 m/s
成功回合接近的约 8 s 预备时间，球与承载板同步从 `x=-0.18 m` 出发，截获线仍为 `x=0.45 m`，
理论到达时间为 7.875 s；传送带上游视觉边界延到 `x=-0.35 m`。承载板半长 0.11 m，因此其初始
左边缘约为 `x=-0.29 m`，仍位于桌面物理范围内。`lift_timeout_steps=1200`、总回合 60 s、7 mm
闭合和 30 mm 净抬升标准均与修正版 G0-S2b 相同。

seed 0 的 MuJoCo 闭环在 10.66 s 达到 `strict_lift_success`，最大净抬升 30.198 mm，最小平移/
旋转误差为 6.959 mm/0.0188 rad。视频逐帧确认板球同步、提前下降、闭合和连续抬升，未再次发生
技能预算耗尽后静止。成果为：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2b_pallet_synced_start_m018_speed008_full_scene_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2b_pallet_synced_start_m018_speed008_timed_frames.png`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2b_pallet_synced_start_m018_speed008_full_scene_cpu.json`

该结果只证明当前固定初态、启用 `grasp_assist` 的 MuJoCo 回合可在 0.08 m/s 下完成；MuJoCo reset
目前没有对初态和动力学做随机化，多 seed 重复并不构成独立稳健性样本。不得据此宣称 Isaac 或真机
0.08 m/s 已稳定通过。下一道有效门禁仍是在物理 GPU1 的独立 Isaac G0-S2b 中同步板球布局，先做
固定 0.08 m/s 多环境严格接触验收，再决定是否从 G0-S1c final warm-start。

### 14.9 Isaac G0-S2b 固定 0.08 m/s 严格验收（2026-10-24）

为在 Isaac 中复现 MuJoCo 的长上游布局，新建独立配置
`config_moving_sphere_g0_s2b_long_entry_speed_008.yaml`。Isaac 的绝对起点由
`entry_x + pallet_start` 组成；保持 `entry_x=0.20 m`，故将板球共同设为 `pallet_start=-0.38 m`，
得到世界坐标 `x=-0.18 m`。托板 USD 的滑轨下限从该配置原有的 -0.12 m 独立扩至 -0.42 m，
上游视觉带延长 0.25 m，且回合完成余量从默认 2 s 增至 10 s。历史配置默认值与场景均不变。
诊断入口新增相同参数并明确记录板球起点，防止再次出现只移动物体而漏移动载具的错误。

从已冻结的 G0-S1c final 开始，不训练，固定速度 0.08 m/s；使用当前 G0-S2a 可选随动参数、
0.10/0.05 PPO 有界残差、1200 步内部抬升预算，并保持 7 mm 闭合、双指接触、8 mm 相对漂移和
真实净抬升 30 mm 标准。4 环境烟雾测试为 0/4，随后正式 256 环境严格验收为：

| 项目 | 结果 |
|---|---:|
| 严格成功 | 0/256 |
| 到达 CLOSE/HOLD/LIFT | 256/256 |
| 曾双指接触 | 0/256 |
| 曾稳定耦合 | 0/256 |
| `cube_exited_zone` | 256/256 |
| 最大真实抬升 | 0 m |
| 最小实时原始平移误差中位数 | 7.075 mm |
| 最小滤波目标平移误差中位数 | 5.360 mm |

逐步轨迹显示，约 7.78 s 时控制器因滤波目标误差 5.82 mm 小于 7 mm 而进入 CLOSE，但实时球心抓取
误差仍为约 8.95 mm；接下来的 16 个闭合步中球以 0.08 m/s 继续前移，夹爪未形成双指接触。
因此提前入场已经解决远距离准备与腕部姿态收敛，当前 0.08 m/s 的真实瓶颈是“滤波目标滞后导致
闭合触发过早 + CLOSE 期间末端未充分同速跟随”，不是回合/抬升步数不足。继续把起点再后移不会
改变这段局部闭合动力学，也不应放宽 7 mm 标准。报告为：

- `artifacts/moving_sphere_static_bowl/diagnostics/g0_s2b_isaac_smoke4_speed008.json`
- `artifacts/moving_sphere_static_bowl/diagnostics/g0_s2b_isaac_eval256_speed008.json`

两次 Isaac 运行均显式设置 `PICKPLACE_PHYSICAL_GPU=1`；运行中通过 GPU UUID 核对进程仅位于物理
GPU1，物理 GPU0 始终约 15 MiB、0% 利用率。未升级驱动，也未修改现有 IsaacLab。

### 14.10 G0-S2c：高速 ALIGN/CLOSE 随动修正与 0.08 m/s 短训（2026-10-24）

针对 14.9 定位出的局部闭合动力学问题，新建独立配置
`config_moving_sphere_g0_s2c_align_follow_speed_008.yaml`，不覆盖 G0-S1c/G0-S2b 结果。修正严格限定在
几何基线和闭合时序：

1. ALIGN 可使用抓取目标的历史 XY 位移作为前馈，使 TCP 在闭合前先与球同速；
2. CLOSE 除滤波抓取误差外，还必须让当前实时原始平移和旋转误差通过同一 7 mm/0.15 rad 门槛，
   防止滞后滤波目标制造提前闭合；
3. CLOSE/HOLD 每步重新计算目标位置反馈并叠加同一几何前馈，8 mm 仅为每步安全上限，不是固定
   位移动作；
4. ALIGN 从球距固定截获线 6 cm 时切入，比 G0-S2b 的 4 cm 提前 2 cm；历史前馈增益固定为 4，
   且绝对限制为 8 mm/步；
5. actor 输入契约不变：仍没有显式传送带/物体速度、接触、力、触觉、电流或扭矩。接触和球位姿
   只用于仿真隐藏奖励、控制器已有几何目标和独立诊断，不作为新增速度传感输入。

重新训练前先做零 PPO 残差几何探针。1.0、1.2 和 4.0 倍前馈且 ALIGN 仍在 4 cm 切入时，最低原始
误差分别约为 7.088、7.081 和 6.885 mm，但都只能连续满足门槛 3 步，未进入 CLOSE。逐步轨迹证明
第 4 步时 TCP 的 Z 仍在完成最后下降，球已继续前移；这不是 7 mm 标准过严，而是 ALIGN 切入仍晚
约一个控制周期。将 ALIGN 提前到 6 cm 后，4 环境探针 4/4 进入 CLOSE/HOLD/LIFT、4/4 双指接触、
4/4 稳定耦合，严格成功 2/4，真实相对初始抬升 28.0--36.4 mm。随后 64 环境纯几何验收为 50/64
严格成功、64/64 双指接触和稳定耦合；恢复冻结 G0-S1c actor 的 0.10/0.05 有界残差后提高到
56/64，仍保持 64/64 接触与耦合。因此满足了“先由几何探针证明接触，再允许 warm-start”的门禁。

关键探针报告为：

- `diagnostics/g0_s2c_geometry_ff1_smoke4_speed008.json`
- `diagnostics/g0_s2c_geometry_ff1p2_smoke4_speed008.json`
- `diagnostics/g0_s2c_geometry_ff4_smoke4_speed008.json`
- `diagnostics/g0_s2c_geometry_alignlead006_ff4_smoke4_speed008.json`
- `diagnostics/g0_s2c_geometry_eval64_speed008.json`
- `diagnostics/g0_s2c_warmstart_residual_eval64_speed008.json`

G0-S2c 随后从冻结的 G0-S1c final 独立 warm-start。训练固定 0.08 m/s、1024 并行环境，总预算仅
1,000,000 帧，每约 250,000 帧保存候选；实际完成 999,424 帧，用时约 3 分钟。训练内首批 962 个
完整回合为 100% 严格成功，但选权重始终使用训练外确定性评估。四个候选的 64 回合结果为：

| checkpoint 帧数 | 严格成功 | 双指接触/稳定耦合 |
|---:|---:|---:|
| 262,144 | 59/64 | 64/64 |
| 507,904 | 61/64 | 64/64 |
| 753,664 | 60/64 | 64/64 |
| 999,424 final | 61/64 | 64/64 |

对并列候选 507,904 与 final 再做 256 回合正式决选：前者为 246/256（96.09%），后者为
235/256（91.80%）；两者均为 256/256 双指接触和稳定耦合，全部失败均为抬升末端超时，追丢、
离带与 TCP 越界均为 0。训练后期存在轻微漂移，因此冻结 507,904 帧节点，而不是机械选择 final：

`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g0_s2c_align_follow_speed_008_1024_no_rtx/checkpoints/ppo_teacher_507904.pt`

正式报告为 `diagnostics/g0_s2c_ppo_teacher_507904_eval256_speed008.json` 和
`diagnostics/g0_s2c_ppo_teacher_final_eval256_speed008.json`。选择理由另存于该 run 的
`selection.json`。后续应先用冻结的 507,904 节点生成全场景 MuJoCo 闭环视频并复核闭合/抬升，
再决定是否增加随机起点和位置扰动；不得为了剩余约 3.9% 失败而放宽 7 mm 或真实 3 cm 标准。

训练与全部 Isaac 评估均显式绑定物理 GPU1（UUID
`GPU-1db4cf3e-527c-ecf9-b149-6b90ab85c840`）。宿主机核对时物理 GPU0 一直为约 15 MiB、0%、
25--26°C；未升级驱动，未修改 `/data/CodePWC/IsaacLab`。

冻结 507,904 帧节点后，补齐了 MuJoCo 桥接层此前缺失的三个 G0-S2c 参数：
`align_target_xy_feedforward`、`close_requires_raw_alignment` 和 `max_close_hold_xy_step`。若不补齐，
虽然视频会加载新权重，却会运行旧 G0-S2b 控制器，不能作为本阶段验证。新增参数透传测试通过后，
使用 Mesa llvmpipe + Xvfb 做纯 CPU 全场景闭环渲染；显式设置 `CUDA_VISIBLE_DEVICES=-1`，未访问
任何物理 GPU。

固定 0.08 m/s、球/托板共同从 `x=-0.18 m` 出发、上游视觉边界 `x=-0.35 m`，MuJoCo 在 9.98 s
达到 `strict_lift_success`。最小平移误差约 0.599 mm，最小旋转误差约 0.0398 rad，最大真实净抬升
30.069 mm；最终处于 LIFT，闭合辅助已触发。全程关键帧复核确认 TCP 朝截获点运动、闭合前与球
同向随动、闭合阶段未停留在旧位置，球保持在两指中间并连续抬升。成果为：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2c_best507904_speed008_full_scene_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2c_best507904_speed008_full_scene_cpu.json`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2c_best507904_speed008_contact_sheet.png`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2c_best507904_speed008_grasp_lift_sheet.png`

本视频保持历史桥接口径 `grasp_assist=true`：只有 TCP 距离和真实夹爪开度同时满足条件后才附着，
因此适合验证策略闭环、截获时序和跨引擎运动趋势；它不等价于纯 MuJoCo 指尖接触迁移成功。

### 14.11 G0-S2c 纯 MuJoCo 接触验收与 G0-S2d 入口（2026-10-24）

为排除 14.10 视频中的辅助附着，MuJoCo 桥接新增了只用于验收的左右指真实碰撞诊断：记录单侧/
双侧接触、最长连续双指接触时间和最小指间距，但这些值不进入 actor。关闭 `grasp_assist` 后，固定
0.08 m/s 的全场景闭环回合仍在 10.18 s 达到严格成功：球净抬升 30.284 mm，双指连续真实接触
2.154 s。视频与报告为：

- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2c_best507904_speed008_pure_contact_probe.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g0_s2c_best507904_speed008_pure_contact_probe.json`

随后使用相同固定工况做 20 回合无渲染重复性验收，结果为 20/20 严格成功、20/20 曾双指接触、
20/20 在结束时仍满足物理稳定夹持，且所有回合 `grasp_assist_attached=false`。报告为
`artifacts/moving_sphere_static_bowl/mujoco/g0_s2c_best507904_speed008_pure_contact_20ep.json`。
这证明当前 MuJoCo 参数在固定工况下不需要先做摩擦或夹爪增益补偿；但 20 个 seed 尚未引入随机
初态，不能冒充泛化测试。

下一独立阶段 G0-S2d 只增加小范围初态扰动：托板关节在 -0.38 m 周围 +/-2 cm，球相对托板的
X/Y 偏移各 +/-1 cm；速度仍固定 0.08 m/s，其余控制、奖励、机器人初态、7 mm 闭合门槛和真实
抬升 3 cm 验收全部冻结。先做零 PPO 残差几何门禁，通过后才从冻结的 G0-S2c 507,904 帧权重
warm-start 短训，避免同时改变多个变量。

G0-S2d 的零残差几何门禁为 58/64 严格成功，64/64 双指接触且 64/64 稳定耦合；冻结的 G0-S2c
507,904 权重在同一批随机初态上为 61/64，证明其受限残差确有帮助。随后从该权重 warm-start
999,424 帧，训练内完整回合显示 100% 成功，但训练外确定性结果为：

| 权重 | 随机初态严格成功 | 双指接触/稳定耦合 |
|---|---:|---:|
| G0-S2c 507,904（训练前基线） | **61/64** | 64/64 |
| G0-S2d 262,144 | 59/64 | 64/64 |
| G0-S2d 507,904 | 59/64 | 64/64 |
| G0-S2d 753,664 | 59/64 | 64/64 |
| G0-S2d final 999,424 | 60/64 | 64/64 |

所有失败均为抬升末端 `time_out`，追丢、离带和 TCP 越界均为 0。由于没有任何新候选超过训练前
基线，不再机械延长训练或对较差候选做 256 回合决选；保留 G0-S2d 全部结果作为对照，主权重和
下一阶段 warm-start 仍冻结为 G0-S2c 507,904。明确选择记录为 G0-S2d run 下的 `selection.json`。

进入完整放置任务前还确认了一个不可跳过的控制接口问题：当前 `StableSphereResidualSkill` 在 LIFT
完成后进入 `DONE`，此时六维机械臂动作被清零且夹爪继续保持闭合。若直接套用历史
`config_moving_sphere_s1_place.yaml`，策略没有机会取得搬运/释放控制权，会复现“抓起后停在原地”的
现象。因此下一阶段应独立命名为 G1：先实现并单测 `LIFT -> bowl hover -> lower -> release` 的明确
控制权交接，以固定静止碗位姿做无速度、无接触输入的几何探针；只有探针能把真实夹持的小球放入
碗内后，才开放受限 PPO 搬运残差并 warm-start。不得直接启动旧 S1 的 6000 万帧长训。

### 14.12 G1 完整放置几何门禁与训练入口（2026-10-25）

新增 `StableSpherePickPlaceSkill`，明确接管 G0 的 `DONE`：先保持抓取位置竖直抬到安全高度，再移动
到静止碗上方、下降并打开夹爪。控制器只读取 6D 抓取候选、TCP、指关节编码器和静止碗位姿；物体/
传送带速度、接触、力、触觉、电流和扭矩仍禁止进入 actor 与控制器。完整任务最初还错误继承了方块
环境的 `4x16` 抓取点模板，导致冻结球策略无法装载；已在 `MovingSphereStaticBowlEnvCfg` 中恢复为
G0 使用的 canonical `40x20` 点云、最佳 6D 位姿差和 40 个候选位姿差。相关单元测试最终为
286/286。

第一版搬运每步位移上限只有 4 mm 命令（实测 TCP 约 0.02 m/s），第 541 步完成抓取、第 791 步才
完成净空，直到第 1079 步仍未到碗，2/2 失败。把搬运上限提高到约 0.08 m/s 并增加命令 slew 后，
固定场景结果从 2/2 烟雾成功扩大为 14/16。两个反例必须保留：继续降速到约 0.06 m/s 后只有
13/16；把连续夹爪命令从已验证的 -0.15 加紧到 -0.35 导致 0/16，原因是额外侧向挤压使球弹出，
所以不能用“更慢/夹得更紧”替代物理对照。

原固定碗中心为 `y=-0.10 m`，距传送带中心 0.40 m。将碗移到安全校验允许的最近实用布局
`y=0.00 m` 后，同 seed 固定场景达到 15/16（93.75%）。尝试 `y=0.05 m` 被实体安全校验拒绝：固定
碗实际外沿与传送带不足 2 cm，未绕过校验。最终 G1 布局固定为 `x=0.45, y=0.00 m`，碗中心训练
随机化为 X/Y 各 +/-1 cm。64 环境随机初态对照为：

| 控制 | 严格完整成功 | 失败类型 |
|---|---:|---:|
| 冻结 G0-S2c PPO 残差 | 50/64（78.13%） | 14 次搬运离带 |
| 零 PPO 残差 | **54/64（84.38%）** | 10 次搬运离带 |

两者均为 64/64 完成 G0 抓取和净抬升，且无超时。因此旧 G0 残差不能原样作为 G1 最终策略；G1
只保留其网络表示做 warm-start，把抓取残差上限减半为每步约 1 mm/0.5 mrad，并由确定性几何控制
独占搬运/释放。新训练配置和入口为：

- `pipeline/0_state_teacher/config_moving_sphere_g1_place_008.yaml`
- `scripts/run_moving_sphere_curriculum_local.sh g1 <G0-S2c-507904.pt>`

配置使用 1024 并行环境、最多 1,000,000 帧、每 250,000 帧独立候选，冻结 actor 方差并加入
action-L1，禁止因训练内成功率高而自动选 final。关键门禁报告位于：

- `diagnostics/g1_geometry_bowl_y000_eval16_fixed_speed008.json`
- `diagnostics/g1_frozen_residual_randomized_eval64_speed008.json`
- `diagnostics/g1_zero_residual_randomized_eval64_speed008.json`

最初的 CUDA 探针和 smoke 报 `cuda_available=False` / `cuInit failed (CUresult 100)`，后来确认不是
GPU1 硬件故障，而是命令运行边界没有授予 GPU 设备访问。改用已授权且显式设置
`PICKPLACE_PHYSICAL_GPU=1` 的启动入口后，64 环境/4096 帧 smoke 正常完成；全程没有切换到物理
GPU0，没有重载或升级驱动，也没有修改 `/data/CodePWC/IsaacLab`。因此以后不能把受限执行环境中的
裸 PyTorch CUDA 探针误判成宿主机设备故障，Isaac 任务必须继续通过同一 GPU1 安全入口启动。

随后从 G0-S2c 507,904 帧权重正式 warm-start G1：1024 个并行环境，共 999,424 帧，用时约
2 分 53 秒，CUDA 峰值约 14.25 GiB。训练内完整回合在约 80 万帧首次出现成功，随后从抓后运输
掉球快速过渡到末段多个批次 99.5%--100% 严格成功；最终权重为：

`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g1_place_008_1024_no_rtx/checkpoints/ppo_teacher_final.pt`

这些数值只证明在线训练批次收敛，不能替代独立确定性评估。验证器也已改为从检查点配置读取
G1 的 0.05/0.025 残差上限，而不是误用 G0 的 0.10/0.05。下一步按相同随机初态分别评估
262,144、507,904、753,664 和 final，按训练外严格成功率选择，不机械采用 final。

四个节点在相同 seed 0 的 64 回合随机初态筛选中分别得到 57/64、56/64、56/64 和 56/64；
所有回合均完成截获、双指抓持和高位净空，失败集中为搬运阶段滑脱。由于一回合差异不足以直接选型，
又以 seed 100 做 256 回合确认：262,144 节点为 221/256（86.33%，34 次离带、1 次超时），final
为 217/256（84.77%，38 次离带、1 次超时）。同一批初态的零残差几何对照也是 217/256，说明短训
前段学到的受限残差带来 4/256 的真实增益，但后续训练已漂回几何基线附近。正式冻结：

`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g1_place_008_1024_no_rtx/checkpoints/ppo_teacher_262144.pt`

完整选择依据记录在同一 run 的 `selection.json`。这次再次证明：训练内末段 99.5%--100% 不能用来
替代训练外确定性评估，尤其并行环境回合在时间轴上成批结束时，末段成功率会被较容易/较晚结束的
样本显著偏置。下一步不延长 G1 PPO，而应先为冻结节点生成完整场景视频，并针对 35/256 个失败
中的搬运滑脱分析初态区域；任何改进都应独立成下一阶段，保留本阶段作为对照。

为避免把旧 G0 视频脚本冒充完整任务，MuJoCo 桥接随后增加了 G1 的明确状态机：G0 抓取完成后依次
执行竖直净空、碗上方横移、下降、打开夹爪，并要求球在碗内释放、相对速度低于 0.05 m/s 且连续
稳定 5 步后才记为成功。该判据使用物体状态只做终止验收，不进入 actor 或控制器。新增桥接和结果
判据单元测试后，相关测试为 32/32。

冻结的 262,144 节点在 MuJoCo、固定 0.08 m/s、`grasp_assist=false` 的首个完整闭环回合中，于
18.08 s 达到 `strict_place_success`。左右指发生真实双侧接触，最长连续双指接触 9.816 s；球最大
净抬升约 160.8 mm，最终距碗中心约 11.8 mm、速度约 0.0143 m/s，并在碗内稳定。渲染使用
Xvfb + Mesa llvmpipe，显式 `CUDA_VISIBLE_DEVICES=-1`，未使用 GPU。完整成果为：

- `artifacts/moving_sphere_static_bowl/mujoco/g1_best262144_speed008_pure_contact_full_scene_cpu.mp4`
- `artifacts/moving_sphere_static_bowl/mujoco/g1_best262144_speed008_pure_contact_full_scene_cpu.json`
- `artifacts/moving_sphere_static_bowl/mujoco/g1_best262144_speed008_pure_contact_contact_sheet.png`

关键帧人工复核确认视频包含 Franka、完整传送带、移动托板、小球和静止碗，运动顺序确为截获—抓取—
抬升—横移—下降—释放，不是骨架或辅助附着演示。这证明固定 seed 的 G1 跨引擎闭环已经打通；它
仍只是首个跨引擎成功回合，下一阶段应做多 seed 纯接触重复性验收，并单独分析失败初态，不能把一条
成功视频外推为 MuJoCo 全随机工况成功率。

随后补充真正改变物理初态的 20-seed 纯接触验收，而不是只更换未参与仿真的随机数：托板世界 X
随机为 -0.20-- -0.16 m，球相对托板 X/Y 各随机 +/-1 cm，静止碗 X 为 0.44--0.46 m、Y 为
+/-1 cm；速度、控制器和判据保持不变。seed 1000--1019 的结果为 **20/20 严格完整成功**，20/20
均曾发生真实双指接触，无滑脱、越界或超时。95% Wilson 成功率区间下界约为 83.89%，因此不能把
20/20 表述成总体必然 100%，但它已经通过了本阶段约定的跨引擎重复性门禁。

实际覆盖到的托板 X 为 -0.19996-- -0.16327 m，球 Y 为 0.290997--0.309452 m，碗 X/Y 为
0.44164--0.45985 m / -0.00939--0.00966 m。完成时间 17.98--18.36 s，首次双指接触时间
7.75--8.39 s，最长连续双指接触 9.614--10.094 s。报告为：

`artifacts/moving_sphere_static_bowl/mujoco/g1_best262144_speed008_pure_contact_randomized_20ep.json`

由于这 20 回合没有失败，无法在 MuJoCo 当前随机范围内识别失败初态簇；Isaac 的 35/256 失败仍
表明跨引擎摩擦、接触和执行器动态存在差异。下一项合理工作是固定策略做 MuJoCo 物理参数扰动，
而不是根据 20/20 继续训练策略或夸大 sim-to-real 结论。

### 14.13 G2 双相机 RGB-D 数据入口（2026-10-25）

G1 262,144 节点保持冻结，下一阶段正式转入高驱动服务器上的双相机 RGB-D 采集。新增了米制深度
观测、G2 数据合同、分级采集 wrapper 和 simulator-free shard 验收器；历史 RGB-only 采集默认值
保持不变。G2 学生只接收双 RGB-D、关节位置、夹爪位置、TCP 位姿、上一动作和标定后的静止碗位姿，
不接收物体/传送带速度、接触、力、触觉、电流或扭矩。`grasp_guidance`、`position_history` 仅保存为
Teacher 标签，动作明确记录为 G1 确定性控制器之前的 PPO 残差。

服务器执行和验收流程见 `docs/tasks/moving_sphere_g2_visual.md`。本机没有启动 RTX 相机；只完成了
4096 帧 smoke 命令的 dry-run 与无仿真测试，因此在服务器真实生成并验证 RGB-D shard 前，不得声称
G2 已经完成采集。

### 14.14 JAKA J0 状态策略起点与初始俯抓姿态（2026-10-25）

JAKA ZU5 + AG95 在本机 GPU1 上的独立状态 Teacher 已开始分阶段训练，不复用 Franka 的完整
actor。所有 Isaac 命令继续显式设置 `PICKPLACE_PHYSICAL_GPU=1`；物理 GPU0 因散热故障禁止使用，
不修改 `/data/CodePWC/IsaacLab`，也不升级驱动。

ZU5 的 reset 关节角不是随意 Home，而是经过几何校验的 shoulder-away 待机解：TCP 初始位置为
`(0.45, 0.30, 0.134) m`，AG95 在策略坐标系中从回合第一帧就是标准竖直朝下的俯抓姿态。
`tcp_rot_xyzw=(0, 0, -0.7071068, -0.7071068)` 用于把 Link_6/AG95 的实体坐标系映射为该策略
TCP，实测初始旋转误差约为 `1e-6 rad`。因此不需要给策略保留一段“低头”时间，也不应
让残差 actor 重新学习这个固定姿态。可视化中偶尔出现的关节构型变化是为了平移和下降，不是 TCP
从水平旋转到朝下。

这一阶段还修正了两个会污染结论的问题：

- actor 的机器人状态现在只包含 6 个 ZU5 关节，AG95 只保留一个真机可获得的逻辑主关节位置；
  其余五个四连杆跟随关节是仿真实现细节，不再输入 actor。
- 早期 J0-S0 评估只显示 29/256，并不是抓取失败，而是 `tcp_workspace_or_retreat` 在成功接近后
  仍以固定截获线判断“连续远离”，将合法抓持抬升误终止为 `tcp_escape`。修正为 near-field
  一旦进入就锁存，只保留全回合硬工作空间检查后，同一个权重重评为 253/256（1.17% 失败，
  **98.83% 严格成功**）。这是终止判据假阴性，不应通过延长训练掩盖。

J0-S0（0.01--0.03 m/s）冻结权重为：

`artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s0a_zero_init_probe500k/checkpoints/ppo_teacher_final.pt`

以它 warm-start 的 J0-S1（0.03--0.05 m/s）500k 短探针已完成。256 回合独立确定性评估为
233/256（**91.02%**），失败为 22 次超时和 2 次球离开传送带，`tcp_escape=0`。final 明显优于
262,144 帧节点的 224/256（87.50%），因此冻结：

`artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt`

下一步不盲目延长 J0-S1，而是先按采样速度和终止阶段分析 22 个超时回合，再决定是调整高速
截获时序，还是直接进入更宽速度课程。JAKA 目前只完成了动态球截获—抓取—抬升，还没有完成
搬运和放入静止碗；在完整状态策略稳定前，不启动高驱动服务器的双 RGB-D 采集。

### 14.15 JAKA 原生 MuJoCo 闭环 sim-to-sim（2026-10-25）

为避免再次把 Isaac 状态轨迹的离线网格回放误称为 sim-to-sim，本阶段独立建立了真正的 JAKA
MuJoCo 桥接。运行时从上传资产构建 ZU5 + AG95 闭链模型、传送带、托板、动态小球和静止碗；
MuJoCo 每个控制步重新产生机器人与物体状态，J0-S1 actor 根据该状态输出残差动作，再经几何截获器、
JAKA 逆运动学和 MuJoCo 执行器推进下一步。因此它不是 Isaac 轨迹回放，也没有运动学逐帧摆拍。

固定 0.04 m/s、seed 31 的首个纯接触闭环回合达到 `strict_lift_success`：球最大净抬升
30.72 mm，左右指均发生真实接触，首次双指接触约 5.538 s，最长连续双指接触约 0.464 s；
`grasp_assist=false`，没有焊接、吸附或抓取辅助。原生画面由 `mujoco.Renderer` 在 Xvfb + Mesa
llvmpipe 上生成，显式 `CUDA_VISIBLE_DEVICES=-1`，不占用 GPU0/GPU1，也未给小球添加轮廓。

- `artifacts/jaka_zu5_ag95_sphere/mujoco/j0_s1_speed004_pure_contact_full_scene_cpu.mp4`
- `artifacts/jaka_zu5_ag95_sphere/mujoco/j0_s1_speed004_pure_contact_full_scene_cpu.json`

对应实现入口为 `mujoco_sim/render_jaka_moving_sphere_native.py`；结果 sidecar 明确记录
`engine=mujoco`、`physics_closed_loop=true`、`isaac_trajectory_replay=false`、
`grasp_assist=false` 和 `sphere_outline_added=false`。该视频用于验证 JAKA 状态策略跨引擎的动作趋势，
但单一固定回合不能替代后续多 seed、速度随机化和完整放置任务验收。

### 14.16 JAKA J0-S2 固定 0.08 m/s 扩展（2026-10-25）

按 Franka 的高速阶段流程，首先冻结 J0-S1 final，在不训练的情况下做固定 0.08 m/s 基线。旧控制器
在 Isaac 中为 0/256：255 次超时、1 次离带。纯几何诊断显示 64/64 都能进入 CLOSE 并产生双指
接触，但 TCP 只抬约 7--9 mm，球只抬约 1--5 mm。因此失败不是截获不到，而是抓后同步随动与
抬升冲突。

直接套用 Franka G0-S2c 参数后，16/16 仍只有约 13.5 mm 平均球抬升。单环境轨迹进一步发现，
控制器把闭合瞬间一帧 XY 目标位移永久锁存；该位移混入了接触瞬间的 Y 扰动，并以过高增益在整个
LIFT 阶段持续积分，导致 TCP 对角漂移约 10 cm、先升后降。这个确定性控制缺陷不能交给 PPO 修补，
因为 residual actor 在 CLOSE/HOLD/LIFT 阶段不拥有动作控制权。

最终的 JAKA 修正保持策略输入合同不变：ALIGN/CLOSE 的前馈仍只来自抓取位姿历史，不输入物体速度；
进入 LIFT 时只保留已知直线传送带 X 轴分量，并在 20 个控制步内平滑衰减到零，滤除接触瞬间横向
扰动且避免永久积分。严格 7 mm 闭合、双指接触、8 mm 耦合漂移和真实表面净空 30 mm 判据均未
放宽。正式纯几何门禁为 64/64 成功、64/64 双指接触，平均接触漂移约 5.11 mm。

恢复冻结的 J0-S1 actor 后，新控制器在 64 回合为 64/64；随后 seed 100 的 256 回合正式独立评估
仍为 **256/256 严格成功**，无超时、离带、TCP 越界或掉球。因此没有启动 J0-S2 PPO 训练：在已经
满成功的确定性边界工况上继续训练没有收益证据，反而可能重现后期策略漂移。J0-S2 的正式冻结对象
是“J0-S1 final 权重 + J0-S2 高速控制配置”组合：

- `artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt`
- `pipeline/0_state_teacher/config_jaka_sphere_j0_s2_speed_008.yaml`
- `artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s1_probe500k/j0_s2_speed008_selection.json`

同一组合随后进入原生 MuJoCo 闭环：固定 0.08 m/s、`controller_profile=j0_s2`、
`grasp_assist=false`，4.14 s 达到 `strict_lift_success`；球净抬升约 30.95 mm，左右指均产生真实
接触，最长连续双指接触约 0.68 s。视频使用 CPU Mesa 渲染，不占用 GPU：

- `artifacts/jaka_zu5_ag95_sphere/mujoco/j0_s2_speed008_pure_contact_full_scene_cpu.mp4`
- `artifacts/jaka_zu5_ag95_sphere/mujoco/j0_s2_speed008_pure_contact_full_scene_cpu.json`

Isaac 全程只使用物理 GPU1；物理 GPU0 未使用，没有修改 `/data/CodePWC/IsaacLab` 或驱动。当前结论
覆盖固定 0.08 m/s 和现有小范围机器人 reset 随机化；下一阶段应先增加球/托板初态扰动和速度邻域
随机化，再进入完整搬运放碗，不能把固定速度的 256/256 外推为所有工况 100%。

### 14.17 JAKA J0-S3 随机化与 J1 完整放碗（2026-10-26）

J0-S3 按单变量课程推进，没有把位置和速度扰动一次性混入。J0-S3a 保持固定 0.08 m/s，只加入托板
起点 +/-2 cm、球相对托板 X/Y 各 +/-1 cm；纯几何闭环 64/64，冻结策略 256/256。随后 J0-S3b
保留同一位置扰动，将速度随机为 0.06--0.08 m/s；实际几何样本覆盖 0.06058--0.07930 m/s，几何
闭环 64/64，冻结策略再次达到 256/256。两阶段都没有启动 PPO，正式冻结对象仍是 J0-S1 final actor
加后续控制配置。选择记录为：

`artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s1_probe500k/j0_s3b_selection.json`

J1 在此基础上增加确定性的竖直净空、移至静止碗、下降和释放状态机；碗中心为 `(0.45, 0.00) m`，
X/Y 各随机 +/-1 cm。AG95 保持已验证的 `close_action=-0.43`，没有误用 Franka 的 -0.15；actor 和
控制器仍不接收传送带/物体速度、接触、力、触觉、电流或扭矩。训练前同种子配对中，零残差与受限
冻结残差分别为 15/16 与 16/16；扩大到 64 回合分别为 62/64 与 63/64，说明旧残差是小幅正贡献。
seed 170 的 256 回合为 248/256，全部 8 次失败都是搬运阶段离带，抓取抬升为 256/256。

随后独立 warm-start 100 万帧 J1 短训：1024 并行环境，实际 999,424 帧，约 3 分 24 秒，峰值 CUDA
分配约 14.24 GiB。训练内后半段虽然频繁出现 98%--100%，四个候选在 seed 180 的 64 回合训练外
筛选却只有 63/64、60/64、61/64、61/64。领先的 262,144 节点在 seed 190 的 256 回合为 247/256；
同 seed 的训练前权重为 **249/256（97.27%）**。因此拒绝全部新训练节点，保留它们作负对照，正式
J1 仍选训练前 actor + J1 几何搬运配置：

- `artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt`
- `pipeline/0_state_teacher/config_jaka_sphere_j1_place_006_008.yaml`
- `artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j1_place_006_008_1024_no_rtx/selection.json`

这次再次验证“训练内成功率不能替代同种子独立评估”。当前剩余失败集中在搬运滑脱，不应回退修改
已经 256/256 的截获抓取阶段。下一步是把正式 J1 组合接入 JAKA 原生 MuJoCo，做完整场景、纯接触的
抓取—搬运—放碗闭环视频与多 seed 验收，再进入高驱动服务器双 RGB-D 采集。

### 14.18 JAKA 原生 MuJoCo J1 完整放碗验收（2026-10-26）

正式 J1 组合已接入 JAKA 原生 MuJoCo 闭环，而不是回放 Isaac 轨迹。抓取段原样复用通过
J0-S2/S3 验收的高速截获、连续闭合和随动抬升；只有抓取状态机到达 DONE 后，才切换到竖直净空、
移至静止碗、下降、开爪和等待球稳定的搬运状态机。目标碗中心设为 `(0.45, 0.00, 0.00) m`，
J1 继续使用冻结的 J0-S1 final actor，平移/旋转残差分别限制为 0.05/0.025，AG95 闭合动作保持
-0.43。actor 不接收物体速度、传送带速度、接触、力、触觉、电流或扭矩。

固定 0.08 m/s、seed 31 的纯物理接触回合在 11.12 s 达到 `strict_place_success`：左右指均发生
真实接触，最长连续双指接触约 7.04 s，球最大净抬升约 18.80 cm；开爪后球在碗内速度约
0.0063 m/s，并连续 5 个控制步满足碗内低速稳定判据。`grasp_assist=false`，没有焊接、吸附或
抓取辅助。完整场景视频由 CPU Mesa 渲染，未占用 GPU0/GPU1，也没有给球添加绿色描边：

- `artifacts/jaka_zu5_ag95_sphere/mujoco/j1_speed008_pick_place_pure_contact_full_scene_cpu.mp4`
- `artifacts/jaka_zu5_ag95_sphere/mujoco/j1_speed008_pick_place_pure_contact_full_scene_cpu.json`

随后做 20 回合真实随机化验收，而不是只更换未被环境消费的随机种子：带速均匀采样
0.075--0.080 m/s，球在托板上的 X/Y 初始位置各扰动约 +/-1 cm，碗 X/Y 标定位置各扰动
+/-1 cm。结果为 **20/20 严格放碗成功**。每个成功都要求曾有真实双指接触、真实抬升、实际
开爪、球位于碗内且速度低于 0.05 m/s 连续 5 步；夹爪开度本身不构成成功。

紧凑验收记录：

`artifacts/jaka_zu5_ag95_sphere/mujoco/j1_randomized_acceptance.json`

该结果证明当前 J1 组合已在 MuJoCo 中完成了 0.08 m/s 邻域的闭环 sim-to-sim。它仍只是小范围
扰动验收，不能外推到任意带速或大范围标定误差。下一阶段可冻结这一成果，再开始高驱动服务器上的
双 RGB-D 并行采集；采集目的是训练视觉观测策略，不改变本阶段无速度输入的状态策略合同。

### 14.19 JAKA J1 放置抖动诊断与稳定化（2026-10-26）

首次 J1 视频虽然严格成功，但人工检查发现机械臂在碗上方下降前明显来回摆动。逐控制步诊断确认，
问题不是视频编码、开爪动作或碗碰撞，而是搬运控制器对加速和减速使用了同一个 0.04/step 的动作
斜率限制。TCP 接近旧 waypoint 后仍携带上一方向的命令进入下一阶段，随后越过新目标再反向修正。
旧版 `BOWL_HOVER` 的 X/Y/Z 动作方向分别切换 11/4/8 次，`LOWER` 为 6/8/6 次；下降阶段 Y
方向活动范围达到约 77.7 mm、耗时 2.54 s。最终开爪阶段其实已经稳定，TCP 每步最大位移仅约
0.056 mm。

修正版保留从静止开始时的加速度限制，但减速命令立即生效；每次
`CLEAR -> BOWL_HOVER -> LOWER` waypoint 交接都清零旧平移命令。下降被明确限制为竖直技能，
仅允许最大 2 mm/step 的小范围 XY 标定修正，并增加 1 mm 位置死区。修正后同一 seed 31 工况：

- `LOWER` 横向活动范围从约 77.7 mm 降至 **1.13 mm**；
- `LOWER` 时间从 2.54 s 降至 **0.56 s**；
- 完整任务时间从 11.12 s 降至 **7.56 s**；
- 开爪阶段继续保持近似静止，并达到 `strict_place_success`。

随后用与 14.18 相同的速度、球起点和碗位姿随机化重新验收，结果仍为 **20/20 严格成功**，说明
稳定化没有用成功率换取观感。完整逐回合记录和新版视频为：

- `artifacts/jaka_zu5_ag95_sphere/mujoco/j1_stabilized_randomized_acceptance_full.json`
- `artifacts/jaka_zu5_ag95_sphere/mujoco/j1_speed008_pick_place_stabilized_full_scene_cpu.mp4`
- `artifacts/jaka_zu5_ag95_sphere/mujoco/j1_speed008_pick_place_stabilized_full_scene_cpu.json`

旧视频保留作负对照，不再作为冻结版本。稳定版通过 63 个相关单元/配置/MuJoCo 回归测试；渲染仍
为 CPU Mesa，`grasp_assist=false`，未占用 GPU0/GPU1。

JAKA 从资产建立、AG95 闭链验证、Isaac 课程、失败诊断、MuJoCo 闭环到未来真机门禁的独立复现文档：

`docs/reproduction/jaka_zu5_ag95_moving_sphere_zh-CN.md`
