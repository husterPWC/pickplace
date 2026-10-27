# JAKA ZU5 + AG95 传送带动态小球抓取放碗：完整实现、复现与问题复盘

> 最后更新：2026-10-26  
> 当前状态：JAKA 状态策略已完成 Isaac 课程、0.08 m/s 高速抓取、位置/速度随机化、静止碗放置，
> 并通过 MuJoCo 纯接触闭环 sim-to-sim。JAKA 双 RGB-D 数据采集、视觉学生和真机闭环尚未执行。  
> 本文是 JAKA 专用冻结记录；更长的逐日实验日志见
> [moving_sphere_static_bowl.md](../tasks/moving_sphere_static_bowl.md)。
> 两个 JAKA 十物体物流任务的统一过程、踩坑和 checkpoint 审计见
> [JAKA 双任务十物体连续处理复盘](jaka_zu5_ag95_ten_item_queue_zh-CN.md)。

## 1. 最终要解决的任务

硬件对象为 JAKA ZU5 六轴机械臂和 DH-Robotics AG95 平行夹爪。小球随传送带沿直线运动，机械臂
需要完成：

```text
等待/截获 → 对齐 → 连续闭合 → 稳定随动 → 抬升 → 竖直净空
→ 搬运至静止碗 → 下降 → 开爪 → 等待小球在碗内稳定
```

当前正式速度域为 `0.06--0.08 m/s`；MuJoCo 最终验收集中在 `0.075--0.08 m/s` 边界邻域。
传送带速度只用于环境随机化，**不是策略输入**。

当前 J1 配置中 `success_requires_home=false`，即严格成功到“小球释放并在碗中稳定”为止，尚未包含
放置后机械臂返回 Home。若要把回 Home 纳入成功条件，应独立建立 J2，不得直接改写当前冻结结果。

### 1.1 同一单球策略连续调用 10 次

已在 MuJoCo 中增加独立的连续调用验证，但没有改变或重新训练冻结 actor。actor 只加载一次；每轮
完成单球抓放后，调度器清空该轮的 10 帧目标历史与抓取状态机，机械臂回到截获准备位，再把下一
个真实刚体小球送入传送带。仿真世界和静止碗不重置，已经放入的小球不删除，并继续参与接触碰撞。

固定起点和 `±8 mm` 随机起点两轮测试均得到：

```text
策略调用成功：       10/10
返回截获准备位：     10/10
最终留在碗中并稳定： 10/10
严格连续任务：       success
```

传送带速度为 `0.08 m/s`，单次策略调用约 `7.24--7.84 s`，回位约 `1.36--1.38 s`。因此当前
保守生产节拍约为 `9 s/球`；这表示可以复用单球策略十次，并不表示机械臂能处理间隔小于一个完整
周期的重叠来球。

评估与视频：

```text
artifacts/jaka_zu5_ag95_sphere/mujoco/
  ten_sphere_repeated_policy_speed008.json
  ten_sphere_repeated_policy_speed008_randomized.json
  ten_sphere_repeated_policy_speed008_full_scene_cpu.mp4
```

对应入口：

```text
mujoco_sim/evaluate_jaka_ten_sphere_sequence.py
mujoco_sim/render_jaka_ten_sphere_sequence.py
```

### 1.2 10 球从起始时刻全部可见的物流队列

在不覆盖 1.1 节逐球送入版本的前提下，另建了更接近物流分拣的连续队列场景。10 个球及其 10 个
独立物理托板在 reset 时全部位于同一条加长传送带上，以 `0.08 m/s` 同步运动；每次任务结束只把
策略关注对象切换到下一颗已经运动中的球，不生成、瞬移或重置任何球，机械臂回位期间整条队列也
持续运动。

根据实测约 `8.7--9.1 s` 的抓放回位周期，球间距取 `0.71 m`。首球初始 `x=0.20 m`，其余球依次
排到 `x=-6.19 m`，传送带范围扩为 `[-6.54, 1.00] m`，总长 `7.54 m`。冻结 actor 仅加载一次，
确定性严格验收结果为：

```text
起始即存在且可碰撞： 10/10
起始同步运动：       10/10
策略调用成功：       10/10
返回截获准备位：     10/10
最终留在碗中并稳定： 10/10
中途刷新/生成球：    false
严格连续任务：       success
```

这里的 actor 是实际冻结网络，不是零残差占位。正式审计版加载 SHA-256 为
`14b1d8af7ecaa46e4a77c48434ca07d43e7d2f3e73672832af663ece2485ea7b` 的 J0-S1 checkpoint；
MuJoCo 每个 `20 ms` 控制步都从当前物理状态构造观测并调用网络。10 轮合计 `3759` 个网络控制步，
`3759/3759` 步输出均非零；各轮平均绝对输出约 `0.00840--0.00860`，最大绝对输出约 `0.02315`。
执行层使用正式 J1 的平移/旋转残差上限 `0.05/0.025`。

对应结果、视频与入口：

```text
artifacts/jaka_zu5_ag95_sphere/mujoco/
  ten_sphere_visible_queue_speed008_checkpoint_closed_loop.json
  ten_sphere_visible_queue_speed008_checkpoint_closed_loop_full_scene_cpu.mp4
  ten_sphere_visible_queue_speed008_checkpoint_closed_loop_full_scene_cpu.json

mujoco_sim/evaluate_jaka_ten_sphere_queue.py
mujoco_sim/render_jaka_ten_sphere_queue.py
```

视频开头使用广角显示整条传送带和全部 10 球，随后切回工作区近景。视频为 MuJoCo 闭环物理、
CPU Mesa 渲染、4 倍速播放，没有添加小球描边。该验收证明当前约 9 秒节拍下的单机械臂串行队列
可行；它不代表球间距可以小于机械臂完整周期，也不代表已完成视觉或真机验收。

## 2. 当前冻结结论

### 2.1 正式权重与配置

正式 actor 不是 J1 短训后的节点，而是 J0-S1 final：

```text
artifacts/jaka_zu5_ag95_sphere/teachers/
  jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt
```

SHA-256：

```text
14b1d8af7ecaa46e4a77c48434ca07d43e7d2f3e73672832af663ece2485ea7b
```

正式完整任务配置：

```text
pipeline/0_state_teacher/config_jaka_sphere_j1_place_006_008.yaml
```

当前配置 SHA-256：

```text
11f4c3a795e8669b831c751b79e3b0ac0fb1a071b14baed9d45aee760b51a146
```

冻结对象必须理解为“**actor 权重 + J0-S2/S3 抓取控制参数 + J1 放置状态机**”，不能只复制 `.pt`
文件。后两部分是动作执行语义的一部分。

### 2.2 最终验收结果

| 阶段 | 环境与扰动 | 严格结果 | 决策 |
|---|---|---:|---|
| J0-S0 | Isaac，0.01--0.03 m/s | 253/256 | 接受 J0-S0 final |
| J0-S1 | Isaac，0.03--0.05 m/s | 233/256 | 接受 J0-S1 final |
| J0-S2 | Isaac，固定 0.08 m/s | 256/256 | 不再训练，接受控制器修正 |
| J0-S3a | Isaac，0.08 m/s + 位置扰动 | 256/256 | 不再训练 |
| J0-S3b | Isaac，0.06--0.08 m/s + 位置扰动 | 256/256 | 不再训练 |
| J1 训练前 | Isaac，完整放碗 | 249/256 | 优于 J1 短训节点 |
| J1 最佳短训节点 | Isaac，同 seed 对比 | 247/256 | 拒绝，保留为负对照 |
| J1 稳定版 | MuJoCo，0.075--0.08 m/s + 初态/碗扰动 | 20/20 | 最终 sim-to-sim 通过 |

最终 MuJoCo 视频：

```text
artifacts/jaka_zu5_ag95_sphere/mujoco/
  j1_speed008_pick_place_stabilized_full_scene_cpu.mp4
```

完整 20 回合记录：

```text
artifacts/jaka_zu5_ag95_sphere/mujoco/
  j1_stabilized_randomized_acceptance_full.json
```

## 3. 不可违反的工程约束

1. 本机 Isaac 只允许使用物理 GPU1。统一通过 `PICKPLACE_PHYSICAL_GPU=1` 运行；GPU0 有散热故障，
   禁止训练、仿真和渲染使用。
2. 不升级 NVIDIA 驱动，不修改 `/data/CodePWC/IsaacLab`。
3. MuJoCo 渲染使用 CPU Mesa：`CUDA_VISIBLE_DEVICES=-1`、`LIBGL_ALWAYS_SOFTWARE=1`、
   `MUJOCO_GL=glfw`，不能占用 GPU0/GPU1。
4. 所有权重、日志、视频和数据都保存在本仓库目录下，不写入其他缓存或用户目录。
5. 新实验使用独立目录，旧最佳权重和失败节点都不覆盖。
6. 训练内回报或训练内成功率不能代替独立、确定性、同 seed 的闭环评估。
7. 不允许通过放宽 7 mm 对齐、夹爪开度或瞬时碰撞，把失败伪装成成功。

## 4. ZU5 + AG95 资产建立

### 4.1 源资产与直接安装

使用的独立 URDF：

```text
assets/models/manipulators/Zu5/zu5.urdf
assets/models/grippers/Ag95Gripper/ag95.urdf
```

组合脚本：

```text
scripts/build_zu5_ag95_urdf.py
```

输出：

```text
assets/models/robots/Zu5Ag95/zu5_ag95.urdf
```

ZU5 `tool0` 与 `ag95_base_link` 使用固定关节直接连接，`xyz=0 0 0`、`rpy=0 0 0`。这是按真机“只有
很薄法兰，可忽略”的实际安装方式建模，不额外虚构长转接件。组合 URDF 当前 SHA-256 为：

```text
f288c9e587d8e8928d33111e4015177e62b0fa723d8e1e11c739d327cf3667e2
```

Isaac 使用转换后的组合 USD：

```text
assets/models/robots/Zu5Ag95/usd/zu5_ag95/zu5_ag95.usda
```

### 4.2 AG95 不是六个独立执行关节

导入模型包含六个连杆关节：

- `left_outer_knuckle_joint`：逻辑主关节；
- 其余五个 knuckle/finger joint：四连杆跟随关节。

真机策略接口只有一个 AG95 位置量。PhysX 不会自动执行转换器写入的 mimic metadata，因此仿真层把
六个连杆坐标同步驱动到同一目标，但 actor 只看到主关节。**不得把五个仿真 follower 当成真机传感器
输入，也不得给策略六维夹爪动作。**

AG95 标定：

- 张开：`q=0`；
- 几何闭合上限：`q=0.943`；
- 正式归一化闭合动作：`close_action=-0.43`，约映射到 `q=0.674`；
- 不能照搬 Franka 的 `-0.15`。

物理验证结果保存在：

```text
artifacts/zu5_ag95_validation/result.json
artifacts/zu5_ag95_validation/zu5_ag95_validation.mp4
artifacts/zu5_ag95_validation/zu5_ag95_gripper_closeup.mp4
```

验证包括开合、物体阻挡、双指接触、带载保持和 follower 耦合。mimic 最大误差约 `0.00186 rad`，
验证总结果为 `passed=true`。

### 4.3 TCP、坐标系和初始俯抓姿态

AG95 两个指垫中点相对工具约为 `167.56 mm`。正式参数：

```text
tcp_offset      = (0, 0, 0.16756)
tcp_rot_xyzw    = (0, 0, -0.7071067812, -0.7071067812)
home_tcp_pos    = (0.45, 0.30, 0.134)
ik_action_scale = 0.02
```

该旋转把 AG95/Link_6 实体坐标变换到策略的规范俯抓坐标：策略 `+y` 对应物理闭合轴，接近方向竖直。
因此机械臂第一帧已经朝下，不需要浪费控制步学习“低头”。

初始关节采用 shoulder-away 解：

```text
[-2.341213, 1.740149, 1.777842, 1.194395, 4.712393, -2.341218]
```

另一个看似自然的 IK 分支虽然达到相同 TCP，但 Link_2 会扫入传送带来球通道，甚至在策略启动前
撞托板/小球。当前分支与来球走廊保留约 138 mm 的保守 AABB 间隙。

### 4.4 Isaac 导入与 IK 的三个陷阱

1. 组合 USD 已带固定 root joint。若再令 spawner 创建固定根，会产生重复约束甚至启动失败；因此
   `fix_root_link=None`。
2. `ag95_base_link` 是固定子体，运行时在 body 列表中存在，但不一定在 reduced Jacobian 中拥有对应
   行。若用它作为 IK body，索引会偏到下一个实体并导致发散。正式 IK body 使用与它重合的
   `Link_6`。
3. 长工具 TCP 不能直接套用短工具的局部 offset Jacobian 公式。正式实现把 167.56 mm offset 旋到
   base 坐标系后使用刚体 Jacobian 关系，否则 TCP 会出现错误方向和漂移。

## 5. 策略与控制器到底是什么

### 5.1 不是纯脚本，也不是端到端纯 PPO

当前方案是“几何粗控制器 + 有界 PPO 残差 + 确定性抓放技能”：

```text
40 个稳定 6D 抓取候选（每个 20 个点）
                    ↓
固定几何截获/对齐控制器 + PPO 小残差
                    ↓
ALIGN → CLOSE → HOLD → LIFT
                    ↓
CLEAR → BOWL_HOVER → LOWER → RELEASE
```

PPO 负责修正截获和接近误差；进入闭合、稳定抬升和搬运状态机后，安全的确定性控制器拥有动作。
这避免让 PPO 从零发现极稀疏的动态抓取时序，也避免把确定性控制器的 bug 错推给 RL 学习。

### 5.2 Actor 输入合同

JAKA actor 使用：

- TCP 位置 `3`；
- TCP 四元数 `4`；
- AG95 主关节位置 `1`；
- ZU5 六个关节相对位置 `6`；
- 上一动作 `7`；
- `40 × 20 × 3` 的 TCP 坐标系抓取点；
- 最近抓取位姿的 6D 误差 `6`；
- `10 × 11` 位置—动作历史，其中每帧为“小球相对 TCP 位置 3 + AG95 位置 1 + 上一动作 7”。

明确不输入：

- 小球线/角速度；
- 传送带速度；
- 关节速度或 TCP 差分速度；
- 接触标志、法向力、力/力矩、触觉、电流或扭矩；
- 五个 AG95 仿真 follower joint；
- Isaac 专用抓取状态。

仿真可以用真实双指接触作为隐藏奖励标签和严格验收诊断，但它不进入 actor。真机不需要新增物体测速、
力或触觉模块。

### 5.3 小球对称性的处理

小球绕球心旋转等价。若每帧重新选择任意 yaw，候选位姿会在等价解之间跳变，策略看到的是人为抖动。
最终使用固定 canonical yaw 和稳定候选选择，保留正确 TCP 高度、接近方向、闭合方向和夹爪宽度；
不能因为球对称就删除全部姿态约束。

## 6. 分阶段课程与关键决策

### 6.1 J0-S0：0.01--0.03 m/s

从验证过的几何控制器开始，actor 输出层严格零初始化，探索方差很小。早期“仅把输出初始化得比较小”
仍会在数百步内积累出单向偏置并触发 TCP 逃逸，因此最终 `actor_out_gain=0.0`。

最初评估只有 29/256，但并非权重突然失效，而是终止器在近场抓取后仍按“离固定截获线越来越远”
判错，把合法抬升终止为 `tcp_escape`。near-field 状态锁存后，同一权重重评为 253/256。这个案例说明
必须先诊断失败类型，不能看到低成功率就盲目延长训练。

冻结权重：

```text
artifacts/jaka_zu5_ag95_sphere/teachers/
  jaka_zu5_ag95_sphere_j0_s0a_zero_init_probe500k/checkpoints/ppo_teacher_final.pt
```

### 6.2 J0-S1：0.03--0.05 m/s

从 J0-S0 warm-start，只扩展速度域。500k 短探针 final 在 256 回合得到 233/256，优于 262,144 帧
节点的 224/256，因此冻结 final。主要失败为 22 次超时、2 次离带，`tcp_escape=0`。

### 6.3 J0-S2：固定 0.08 m/s

直接把旧控制器放到 0.08 m/s 得到 0/256，但纯几何诊断显示 64/64 都已经闭合并产生双指接触；
TCP 只抬 7--9 mm、球只抬 1--5 mm。真正问题是抓后随动，不是“追不上”或 7 mm 对齐标准太严。

从视频发现球在机械臂完成悬停后已经接近离开抓取窗口，因此先把下降触发距离从约 8 cm 提前到
16 cm，并在 ALIGN/CLOSE 中保持随动。之后又发现 LIFT 把闭合瞬间的一帧 XY 位移永久锁存，其中
混入 Y 接触扰动，导致 TCP 对角漂移约 10 cm、先升后降。

最终修正：

- ALIGN/CLOSE 使用抓取目标历史产生几何前馈；
- CLOSE 同时检查实时原始误差和滤波误差；
- CLOSE 期间继续随动，不停在旧位置；
- LIFT 只保留已知直线传送带的 X 分量；
- 前馈在 20 控制步内衰减到零；
- 球—TCP 相对 XY 可在真实抬升后重新锁存。

严格 7 mm 对齐、真实双指接触、8 mm 耦合漂移和净抬升 30 mm 标准均未放宽。几何门禁 64/64，
冻结 actor 的 Isaac 评估 256/256，因此**没有启动 J0-S2 PPO**。在满成功的边界工况继续训练只会
增加策略漂移风险。

### 6.4 J0-S3：一次只扩展一个随机化维度

J0-S3a 固定 0.08 m/s，只加入：

- 托板起点 `±2 cm`；
- 小球相对托板 X/Y 各 `±1 cm`。

几何 64/64、策略 256/256。J0-S3b 再将速度扩为 `0.06--0.08 m/s`，仍为几何 64/64、策略
256/256。两阶段均未训练新 actor。

选择记录：

```text
artifacts/jaka_zu5_ag95_sphere/teachers/
  jaka_zu5_ag95_sphere_j0_s1_probe500k/j0_s2_speed008_selection.json
  jaka_zu5_ag95_sphere_j0_s1_probe500k/j0_s3b_selection.json
```

### 6.5 J1：搬运并放入静止碗

J1 在已验证的抓取后增加 `CLEAR → BOWL_HOVER → LOWER → RELEASE → DONE`。碗中心为
`(0.45, 0.00, 0.00) m`，训练/评估随机化 X/Y 各 `±1 cm`。搬运阶段不重新把接触或速度送给 actor。

训练前冻结残差在 64 回合为 63/64，256 回合为 249/256。随后做了 999,424 帧 J1 warm-start
短训，1024 并行环境，约 204 秒，峰值显存约 14.24 GiB。训练内后半程常见 98%--100%，但独立
同 seed 验收中，最佳 262,144 帧节点只有 247/256，低于训练前权重的 249/256。因此全部 J1 新
checkpoint 被拒绝，仅作为负对照保留。

这是本项目最重要的模型选择案例：**训练不是默认有益操作；独立评估没有提升时，应保留旧 actor，
只接受经过验证的控制器扩展。**

## 7. MuJoCo 闭环 sim-to-sim

### 7.1 什么才算闭环 sim-to-sim

早期网格视频只是把 Isaac 记录的状态重新绘制，能检查外观，但不证明策略跨引擎。正式桥接位于：

```text
mujoco_sim/jaka_scene.py
mujoco_sim/jaka_moving_sphere.py
mujoco_sim/evaluate_jaka_moving_sphere.py
mujoco_sim/render_jaka_moving_sphere_native.py
```

每个 20 ms 控制周期由 MuJoCo 产生新的 TCP、关节、夹爪、小球和接触状态，actor 根据新状态输出，
再由 MuJoCo 执行动作推进下一步。这才是闭环 sim-to-sim，不是轨迹回放或逐帧摆拍。

MuJoCo 中 `grasp_assist=false`，没有焊接、吸附或隐式抓取约束；严格成功需要真实左右指接触历史、
真实小球抬升、开爪、小球进入碗并以低于 `0.05 m/s` 的速度连续稳定 5 个控制步。

为避免网格 hull 产生与传送带的虚假碰撞，MuJoCo 桥接关闭了机械臂 link mesh 的物理碰撞，但保留
AG95 指垫/连杆、小球、托板和碗的实际接触。因此该桥接适合验证操作接触和控制趋势，但不能替代
完整机械臂碰撞安全验证。

### 7.2 0.08 m/s 与最终放置

J0-S2 在 MuJoCo 固定 0.08 m/s、纯接触条件下 4.14 s 达到 `strict_lift_success`，净抬升约
30.95 mm。J1 首版在 11.12 s 达到 `strict_place_success`，随机化 20/20，但视频暴露出明显的放碗
前晃动。这说明“任务成功”不等于“运动质量合格”，必须人工看视频并记录轨迹。

### 7.3 放置抖动的根因与修正

旧控制器对加速和减速使用同一个 `0.04/action-step` 斜率限制。TCP 到达旧 waypoint 后，上一阶段的
横向命令仍被带入下一阶段，机械臂越过碗中心再反向修正：

- 旧 `LOWER` Y 活动范围约 77.7 mm；
- 旧 `LOWER` 耗时 2.54 s；
- 动作方向多次反转；
- 真正开爪阶段其实已经稳定。

最终修正：

- 只限制从静止开始的加速，减速命令立即生效；
- 每个 waypoint 切换时清零旧平移命令；
- LOWER 明确为竖直技能，XY 仅允许最大 2 mm/step 的标定修正；
- 增加 1 mm 位置死区。

同一 seed 31 修正后：

| 指标 | 旧版 | 稳定版 |
|---|---:|---:|
| LOWER 横向活动范围 | 77.7 mm | 1.13 mm |
| LOWER 时间 | 2.54 s | 0.56 s |
| 完整任务时间 | 11.12 s | 7.56 s |
| 随机化严格成功 | 20/20 | 20/20 |

旧视频保留为负对照；正式冻结视频是文件名包含 `stabilized` 的版本。

## 8. 全部主要踩坑与防复发规则

| 症状/错误结论 | 真实原因 | 修复 | 防复发规则 |
|---|---|---|---|
| AG95 URDF 中多个关节都能单独动 | 四连杆 follower 被当成独立 DOF | 一维逻辑主关节，同步驱动 follower | actor 永远只看 AG95 主编码器 |
| 夹爪自由开合正常，接触后逻辑异常 | mimic metadata 在 PhysX 未自动生效或 follower drive 对抗约束 | 明确耦合控制并做带载验证 | 必须做阻挡、接触、带载保持，不只看空载动画 |
| 可视化只有骨架 | 使用了线框/关节轨迹而非 mesh renderer | 用真实 STL/USD mesh 回放或 MuJoCo 原生 renderer | “能动”与“完整模型可视化”分开验收 |
| IK 静止也漂或方向不对 | 固定子体 Jacobian 行、长 TCP offset 坐标公式错误 | 选 Link_6，并修正 TCP Jacobian | 新机械臂先跑 idle、单轴、TCP 数值测试 |
| 初始机械臂会碰传送带 | 选了错误 IK 分支 | shoulder-away standby | 同 TCP 的 IK 解也要检查整个 link swept volume |
| 策略开局先低头，浪费动态窗口 | Home TCP 姿态未规范化 | reset 即标准俯抓 | 固定姿态不交给 RL 重学 |
| 小球候选姿态乱转 | 轴对称物体存在无限等价 yaw | canonical yaw + 稳定目标选择 | 对称性必须显式消歧 |
| 训练很久仍朝反方向运动 | 早期纯 RL 奖励/动作尺度无法提供可靠粗运动 | 100/100 几何截获器 + 有界 PPO 残差 | 先单元测试朝目标一步奖励更高，再训练 |
| J0-S0 看似只有 29/256 | 终止器把合法近场抬升误判为远离截获线 | near-field 锁存，保留硬工作区 | 低成功率先按 termination 分类，不盲训 |
| 0.08 m/s 0/256 | 抓到了但抓后随动/抬升冲突 | 提前下降、闭合继续随动、LIFT 前馈衰减 | 先区分“未接触”和“已接触但抬不起” |
| 直接套 Franka 参数仍漂移 | ZU5/AG95 的动力学、TCP 和夹爪时序不同 | 针对 JAKA 重标定 | 机械臂迁移复用流程，不复制控制数值 |
| 闭合瞬间后对角漂移 10 cm | 锁存一帧含 Y 噪声的 XY 位移并永久积分 | 只保留传送带 X 前馈并衰减 | 不把瞬时差分当长期速度指令 |
| 夹爪开度显示“成功” | 中间开度可能只是闭合过程或碰撞 | 必须真实抬升、耦合、双指隐藏标签 | 夹爪开度不是成功条件 |
| 训练内 100% 就选 final | 同分布在线统计乐观，策略可能后期漂移 | 独立同 seed 比 checkpoint | 模型选择只看独立严格评估 |
| J1 再训练反而下降 | 已有残差足够，训练改变了有效行为 | 拒绝 247/256，保留 249/256 pretrain | 训练必须证明增益，不是默认步骤 |
| MuJoCo 视频成功但机械臂放碗前晃 | waypoint 之间继承旧速度命令 | 非对称加减速、阶段清零、竖直下降 | 验收必须包含轨迹质量与视频，不只看 outcome |
| 把网格回放称为 sim-to-sim | 物理状态仍来自 Isaac | 每步由 MuJoCo 生成状态并闭环调用策略 | sidecar 必须记录 `physics_closed_loop=true` |
| 为看视频想升级驱动 | 本机 RTX renderer 与旧驱动不兼容 | Isaac 无相机训练；MuJoCo CPU Mesa 单独渲染 | 不为可视化破坏已部署 IsaacLab/驱动 |
| GPU0 掉卡 | GPU0 散热故障 | Isaac 强制物理 GPU1；MuJoCo CPU | 每条本机 Isaac 命令显式 GPU 映射 |

## 9. 复现命令

### 9.1 Isaac 课程入口

```bash
cd /data/CodePWC/pickplace
PICKPLACE_PHYSICAL_GPU=1 scripts/run_jaka_sphere_curriculum_local.sh j0s0

PICKPLACE_PHYSICAL_GPU=1 scripts/run_jaka_sphere_curriculum_local.sh j0s1 \
  artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s0a_zero_init_probe500k/checkpoints/ppo_teacher_final.pt
```

J0-S2/S3 当前已经通过几何门禁和冻结 actor 评估，除非环境或控制器发生实质变化，不应机械重训。

### 9.2 MuJoCo 单回合严格评估

```bash
xvfb-run -a env \
  CUDA_VISIBLE_DEVICES=-1 \
  LIBGL_ALWAYS_SOFTWARE=1 \
  MUJOCO_GL=glfw \
  .venv/bin/python mujoco_sim/evaluate_jaka_moving_sphere.py \
  controller_profile=j1 pick_place=true belt_speed=0.08 max_seconds=30 \
  seed=31 grasp_assist=false
```

### 9.3 MuJoCo 随机化验收

```bash
xvfb-run -a env \
  CUDA_VISIBLE_DEVICES=-1 \
  LIBGL_ALWAYS_SOFTWARE=1 \
  MUJOCO_GL=glfw \
  .venv/bin/python mujoco_sim/evaluate_jaka_moving_sphere.py \
  controller_profile=j1 pick_place=true belt_speed=0.08 max_seconds=30 \
  seed=3100 episodes=20 randomize=true grasp_assist=false \
  report_out=artifacts/jaka_zu5_ag95_sphere/mujoco/j1_stabilized_randomized_acceptance_full.json
```

### 9.4 稳定版视频

```bash
xvfb-run -a env \
  CUDA_VISIBLE_DEVICES=-1 \
  LIBGL_ALWAYS_SOFTWARE=1 \
  MUJOCO_GL=glfw \
  .venv/bin/python mujoco_sim/render_jaka_moving_sphere_native.py \
  controller_profile=j1 pick_place=true belt_speed=0.08 max_seconds=30 \
  seed=31 grasp_assist=false \
  out=artifacts/jaka_zu5_ag95_sphere/mujoco/j1_speed008_pick_place_stabilized_full_scene_cpu.mp4
```

## 10. 当前真机输入与未来视觉阶段

### 10.1 当前状态策略的真机等价输入

若直接复现当前几何 actor，需要：

- ZU5 六关节位置和 FK 得到的 TCP 位姿；
- AG95 单一主关节位置反馈；
- 上一动作；
- 感知模块输出并转换到 AG95 TCP 坐标系的 40 个 6D 抓取候选/点集；
- 最近 10 帧的位置—动作历史；
- 标定后的静止碗位姿供 actor 外部的确定性放置状态机使用。

不需要速度、力、触觉、电流或双指接触传感器。

### 10.2 高驱动服务器要做什么

高驱动服务器用于 JAKA 双 RGB-D 并行 rollout 和视觉学生数据，不是为了重新训练当前状态 Teacher。
计划相机为固定俯视 RGB-D + 腕部 RGB-D。每步保存：

- 两路 RGB/深度；
- 关节、TCP、AG95 编码器和上一动作；
- Teacher residual action；
- 抓取候选/几何标签；
- 奖励、终止和失败类型。

视觉采集与当前抓取位姿逻辑不冲突：几何状态 Teacher 产生监督，视觉学生从图像学习可部署观测。
学生不得直接读取仿真隐藏速度、接触或精确物体状态。如果以后希望真机只保留单腕部相机，必须在
双相机数据上做独立消融并重新验收，不能在未验证时口头删除固定相机。

## 11. 尚未完成，不得误报

- JAKA 高驱动服务器双 RGB-D smoke/pilot/full 数据采集；
- JAKA 视觉学生训练与 Isaac 视觉闭环；
- 摄像机内参、腕部手眼外参和时间同步的真机标定；
- JAKA 真机低速 dry-run、软限位、急停和失败自动停机；
- 由静止球到慢速带、再到 0.08 m/s 的真机课程；
- 放碗后返回 Home 的 J2；
- 大范围速度、光照、材质、相机外参和传送带标定误差鲁棒性；
- 全机械臂 mesh 碰撞安全的跨引擎验证。

所以当前准确表述是：**JAKA 状态策略和纯接触 MuJoCo sim-to-sim 已走通；视觉 sim-to-real 与真机
闭环尚未完成。**

## 12. 下一阶段强制门禁

1. 冻结本文列出的 actor、J1 配置和稳定版 MuJoCo 结果。
2. 高驱动服务器先做 4096 帧双 RGB-D smoke，检查黑屏、深度单位、字段和动作语义。
3. pilot 数据检查遮挡、时间同步和抓取候选稳定性，通过后才做多 seed full。
4. 视觉学生训练后先做 Isaac 视觉闭环，不以离线 loss 代替成功率。
5. 保持状态 Teacher 输入合同不变，重新跑 JAKA MuJoCo 接口测试。
6. 真机先空载和静止球，再低速传送带，最后逐步到 0.08 m/s。
7. 任一阶段失败都按“未截获、未双指、抓后滑脱、搬运滑脱、落碗失败、超时、安全终止”分类，
   不直接用更多训练步数覆盖问题。

## 13. 关键文件索引

| 内容 | 路径 |
|---|---|
| JAKA arm plug-in | `pickplace/arms/zu5_ag95.py` |
| 长 TCP IK/Jacobian | `pickplace/actions.py` |
| 组合 URDF 工具 | `scripts/build_zu5_ag95_urdf.py` |
| AG95 物理验证 | `scripts/validate_zu5_ag95.py` |
| 本机课程入口 | `scripts/run_jaka_sphere_curriculum_local.sh` |
| 抓取/放置技能 | `pickplace/grasp_skills.py` |
| J0-S0 配置 | `pipeline/0_state_teacher/config_jaka_sphere_j0_s0_residual_001_003.yaml` |
| J0-S1 配置 | `pipeline/0_state_teacher/config_jaka_sphere_j0_s1_residual_003_005.yaml` |
| J0-S2 配置 | `pipeline/0_state_teacher/config_jaka_sphere_j0_s2_speed_008.yaml` |
| J0-S3a 配置 | `pipeline/0_state_teacher/config_jaka_sphere_j0_s3a_position_randomization_008.yaml` |
| J0-S3b 配置 | `pipeline/0_state_teacher/config_jaka_sphere_j0_s3b_speed_006_008.yaml` |
| J1 配置 | `pipeline/0_state_teacher/config_jaka_sphere_j1_place_006_008.yaml` |
| MuJoCo 场景 | `mujoco_sim/jaka_scene.py` |
| MuJoCo 闭环 | `mujoco_sim/jaka_moving_sphere.py` |
| MuJoCo 评估 | `mujoco_sim/evaluate_jaka_moving_sphere.py` |
| MuJoCo 渲染 | `mujoco_sim/render_jaka_moving_sphere_native.py` |
| 10 球可见队列评估 | `mujoco_sim/evaluate_jaka_ten_sphere_queue.py` |
| 10 球可见队列渲染 | `mujoco_sim/render_jaka_ten_sphere_queue.py` |
| 总实验日志 | `docs/tasks/moving_sphere_static_bowl.md` |
