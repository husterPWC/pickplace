# JAKA ZU5 + AG95 双任务十物体连续处理：实现过程、问题与复现记录

> 最后更新：2026-10-26  
> 记录范围：MuJoCo 中的十物体连续物流验证，不替代单回合 Isaac 训练记录。  
> 当前结论：两个任务均已由冻结 checkpoint 逐控制步闭环推理并严格完成 `10/10`；视觉策略、真机节拍与异常恢复尚未验收。

本文集中记录以下两个任务如何从“单次策略成功”扩展为“十个物体从初始时刻全部存在、持续运动、连续处理”的物流队列，以及过程中出现过的错误和修正：

1. **任务 A：**10 个动态小球依次抓取并放入同一个静止碗；
2. **任务 B：**10 个静止小球依次抓取并放入 10 个动态碗。

各任务的完整训练过程、单回合控制细节与资产说明仍以以下文档为准：

- [动态小球 → 静止碗](jaka_zu5_ag95_moving_sphere_zh-CN.md)
- [静止小球 → 动态碗](jaka_zu5_ag95_static_sphere_moving_bowl_zh-CN.md)

## 1. 最终状态一览

| 项目 | 任务 A：动态球 → 静止碗 | 任务 B：静止球 → 动态碗 |
|---|---|---|
| 机器人 | JAKA ZU5 + AG95 | JAKA ZU5 + AG95 |
| 初始物体 | 10 球、10 独立托板，全部可见且运动 | 2×5 供料架中的 10 个静止球；10 碗、10 独立托板全部可见且运动 |
| 目标速度 | 球/托板 `0.08 m/s` | 碗/托板 `0.10 m/s` |
| 正式间距 | `0.71 m` | `0.75 m` |
| 传送带范围 | `[-6.54, 1.00] m` | `[-7.02, 8.13] m` |
| 传送带长度 | `7.54 m` | `15.15 m` |
| 世界重置 | 十次调用之间不重置 | 十次调用之间不重置 |
| 中途生成/瞬移 | 无 | 无 |
| 已完成物体 | 留在静止碗中，继续参与接触 | 随各自碗继续向下游运动，继续参与接触 |
| 严格结果 | 调用 `10/10`、回位 `10/10`、碗内稳定 `10/10` | 抓取—动态放碗—Home `10/10`、对应碗内稳定 `10/10` |
| 推理方式 | 冻结网络残差 + 几何技能主干 | 冻结网络残差 + 几何技能主干 |
| 引擎/渲染 | MuJoCo 物理闭环；CPU Mesa 视频 | MuJoCo 物理闭环；CPU Mesa 视频 |

这里的“策略”不是端到端网络独自生成整段轨迹，而是项目已经冻结的**混合策略系统**：

```text
当前 MuJoCo 物理状态
        ↓
冻结 GRU actor 每 20 ms 输出 7 维有界残差
        +
几何抓取目标、阶段状态机和 IK 主干
        ↓
关节/夹爪控制 → MuJoCo 接触动力学 → 下一时刻状态
```

因此它是闭环 sim-to-sim，不是轨迹回放；但也不能表述成“神经网络单独控制了全部运动”。几何主干、状态机、残差尺度与 checkpoint 一起构成可部署控制器。

## 2. 从单球成功到真实十球队列

### 2.1 第一版：把单球策略调用十次

最先验证的是最低风险问题：同一个冻结 actor 是否能在同一 MuJoCo 世界中重复调用十次。actor 只加载一次，每次完成抓放后清空目标历史和技能阶段，机械臂返回截获准备位，再把下一颗球送入场景。静止碗不重置，已经放入碗中的球不删除。

固定起点和 `±8 mm` 随机起点都完成：

```text
策略调用：       10/10
返回准备位：     10/10
最终碗内稳定：   10/10
```

这一版证明了“单球控制器可以被调度十次”，但没有证明真实物流队列能力，因为后续球是按调用逐个送入的，画面上不是十球从一开始就排在同一条传送带上。用户指出这一点后，该版被保留为**重复调用基线**，不再作为最终十球物流结论。

对应产物：

```text
artifacts/jaka_zu5_ag95_sphere/mujoco/
  ten_sphere_repeated_policy_speed008.json
  ten_sphere_repeated_policy_speed008_randomized.json
  ten_sphere_repeated_policy_speed008_full_scene_cpu.mp4
```

### 2.2 第二版：十球和十块托板从 reset 起全部存在

最终版本为每个球建立独立自由刚体和独立物理托板。reset 时十个球已经按队列排列，全部具有碰撞并以 `0.08 m/s` 同步运动。十次调用之间只切换“当前关注的球”，不会创建、瞬移或复位任何球。

关键实现原则是使用**全局世界时间**推进所有托板。机械臂抓放和回位期间，不只是当前目标球，其余九个托板也必须继续运动。否则后续球会在回位阶段被暗中冻结，得到不真实的额外等待时间。

单球完整抓放加回位的实测周期约为 `8.7--9.1 s`。以 `0.08 m/s` 计算，队列间距至少需要覆盖一个服务周期：

```text
0.08 m/s × 约 8.9 s ≈ 0.71 m
```

因此正式间距取 `0.71 m`。首球初始 `x=0.20 m`，第十球约为 `x=-6.19 m`；传送带向上游扩展到 `-6.54 m`，下游为 `1.00 m`，总长 `7.54 m`。

正式评估还检查了：十球起始时都存在并可碰撞、十球起始速度正确、调用之间无 spawn/reset、已放球未删除、十次回位成功、最终十球都留在碗内并稳定。

### 2.3 任务 A 的冻结网络审计

正式 checkpoint：

```text
artifacts/jaka_zu5_ag95_sphere/teachers/
  jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt
```

SHA-256：

```text
14b1d8af7ecaa46e4a77c48434ca07d43e7d2f3e73672832af663ece2485ea7b
```

网络在每个 `20 ms` 控制步根据当前 MuJoCo 状态重新推理。十轮共 `3759` 个网络控制步，`3759/3759` 步输出非零；每轮平均绝对动作约 `0.00840--0.00860`，全程最大绝对动作约 `0.02315`。执行层沿用 J1 平移/旋转残差上限 `0.05/0.025`。

正式结果和视频：

```text
artifacts/jaka_zu5_ag95_sphere/mujoco/
  ten_sphere_visible_queue_speed008_checkpoint_closed_loop.json
  ten_sphere_visible_queue_speed008_checkpoint_closed_loop_full_scene_cpu.mp4
  ten_sphere_visible_queue_speed008_checkpoint_closed_loop_full_scene_cpu.json
```

视频先用广角证明十球在起始时刻全部存在，再切换工作区近景；播放为 4 倍速，没有给小球添加绿色描边。

## 3. 十个静止球分入十个动态碗

### 3.1 场景结构

十个球从 reset 起就放在静止低沿供料架的 `2×5` 阵列中。十个碗分别固定在十块独立物理托板上，从上游按队列排列并以 `0.10 m/s` 同步运动。第 `i` 次调用选择第 `i` 个球和第 `i` 个碗；世界不重置，成功后的球留在指定碗中并随碗继续向下游移动。

相比任务 A，这里不仅要为未来目标预留上游长度，还要为已经装球的碗预留下游长度。如果传送带只向上游加长，早期装满的碗会在第十次调用之前驶离支撑面，最终保留判定就失去物理意义。

### 3.2 第一次失败：`0.63 m` 间距只完成 3 个

初版根据单球从中心供料点抓取的 `6.2--6.32 s` 周期，按 `0.10 m/s` 估算间距：

```text
0.10 m/s × 6.3 s = 0.63 m
```

但十球版本的球位于 `2×5` 阵列，不同球到机器人基座的距离和 IK 路径不同，真实单次周期增至 `6.9--7.44 s`。到第四次调用时，目标碗已经前进到约 `x=0.25 m`；机器人完成抓球后才进入碗跟踪，剩余追踪窗口不足，最终发生 `sphere_off_table`。严格执行因此停在 `3/10`。

该问题不是“网络不会抓”，也不是应该放宽成功标准，而是**调度节拍错误**：用单一中心点的平均周期替代了队列中最慢供料位的周期。修正时没有提前判成功、没有放宽碗内距离，也没有把失败球吸附进碗。

正式间距按实测最长周期上取整为 `0.75 m`。同时把传送带两端都延长到 `[-7.02, 8.13] m`，总长 `15.15 m`。修正后十个球全部完成抓取、放入对应动态碗、返回 Home，并在最终结算时仍保持在各自碗中。

### 3.3 第二次错误：零残差几何基线被误称为网络能力

第一段成功视频虽然是真实 MuJoCo 接触闭环，但 evaluator/render 当时向控制器传入的是 `np.zeros(7)`。它证明的是“几何状态机 + 零残差”可以完成十物体队列，并不能证明冻结 actor 已经在每步参与。

这曾被错误表述成“实际策略网络能力”。发现后采取了以下修正：

1. 保留原结果，但把文件明确改名为 `zero_residual_baseline`，不删除负面过程证据；
2. evaluator 和 renderer 显式构造 `GraspPointGRUActor(checkpoint)`；
3. 每个 `20 ms` 控制步使用当前观测执行 `actor.act(observation)`；
4. 报告 checkpoint 绝对路径、SHA-256、残差上限、网络步数、非零步数和动作幅值；
5. 重新完成十轮严格评估并重新渲染带 `checkpoint_closed_loop` 的正式视频；
6. 正式结论只引用 `checkpoint_closed_loop`，不再引用旧的无标签文件。

这是本阶段最重要的审计教训：**视频看起来正确，并不自动证明神经网络参与了控制。**必须从程序路径和报告同时证明 checkpoint 被加载、网络逐步被调用、输出不是常数零，并记录其动作约束。

### 3.4 任务 B 的冻结网络审计

正式 checkpoint：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/
  jaka_static_sphere_moving_bowl_m0_final_geometry_smoke16/
  checkpoints/ppo_teacher_final.pt
```

SHA-256：

```text
d522d818dccdb7a073708b75ccb44fb7bc9b98a5780e10f057b5faba3b01555b
```

十轮共 `3526` 个网络控制步，`3526/3526` 步输出非零；各轮平均绝对动作约 `0.00320--0.00390`，全程最大绝对动作约 `0.01357`。执行层使用 M0 的平移/旋转残差上限 `0.03/0.015`。

正式 checkpoint 闭环产物：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/mujoco/
  ten_static_spheres_ten_visible_moving_bowls_speed010_checkpoint_closed_loop.json
  ten_static_spheres_ten_visible_moving_bowls_speed010_checkpoint_closed_loop_full_scene_cpu.mp4
  ten_static_spheres_ten_visible_moving_bowls_speed010_checkpoint_closed_loop_full_scene_cpu.json
```

保留但不能作为网络能力证据的零残差基线：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/mujoco/
  ten_static_spheres_ten_visible_moving_bowls_speed010_zero_residual_baseline.json
  ten_static_spheres_ten_visible_moving_bowls_speed010_zero_residual_baseline_full_scene_cpu.mp4
  ten_static_spheres_ten_visible_moving_bowls_speed010_zero_residual_baseline_full_scene_cpu.json
```

## 4. 严格验收标准

两个任务都不能仅凭“夹爪闭合”“球曾经抬起”“球曾进入碗上方”或“视频看起来差不多”判成功。

### 4.1 共同要求

- 十个目标从 reset 起全部是真实、可碰撞刚体；
- 十次调用之间不重置世界，不中途生成、瞬移或删除目标；
- 所有传送带载体在抓取、放置和回位期间连续运动；
- 无 grasp assist、无隐式吸附；
- 网络在每个控制步接收当前 MuJoCo 状态并输出动作；
- 十次调用全部达到对应单回合严格终止条件；
- 最后额外沉降，再检查所有球是否仍在目标容器内且相对目标稳定。

### 4.2 任务 A 的额外要求

- 每轮小球必须被真实双指夹持、净抬升并释放进同一个静止碗；
- 机械臂每轮必须返回下一次截获准备位；
- 第十轮结束时十个球都保留在静止碗中。

### 4.3 任务 B 的额外要求

- 每轮必须选择唯一的静止球—动态碗配对；
- 成功终止包含真实抬升、随动放碗、相对碗沉降、撤离和 Home；
- 最终用球相对对应碗的位置和速度判断稳定，不能用世界绝对速度。碗以 `0.10 m/s` 运动时，稳定在碗里的球也具有约 `0.10 m/s` 世界速度。

## 5. 踩坑与迁移教训

| 现象/错误 | 根因 | 修正 | 后续 JAKA/真机必须记住 |
|---|---|---|---|
| 十次都成功，但画面像逐个刷新球 | 后续球按调用才被送入，未构成真实队列 | 建立十个独立球和托板，reset 时全部存在 | 区分“重复调用能力”和“物流队列能力” |
| 机械臂回位时后续物体获得额外等待 | 只更新当前载体，队列其余对象被冻结 | 所有托板按全局世界时间持续推进 | 调度器不能暂停真实传送带来迁就策略 |
| 十动态碗使用 `0.63 m` 后第四轮失败 | 间距来自中心球平均周期，忽略 2×5 供料位差异 | 用最慢实测周期确定 `0.75 m` 间距 | 节拍设计使用尾部延迟/最坏周期，不只用均值 |
| 早期装满的碗可能离开支撑面 | 只考虑上游候选队列，没有考虑下游累积 | 传送带双向扩展至 15.15 m | 场景长度要覆盖“等待 + 服务 + 已完成物下游运输” |
| 成功视频被称为网络策略证明 | 实际传入 `np.zeros(7)`，只有几何主干在工作 | 基线改名；逐步加载 actor；记录 hash 和动作审计 | 控制证据必须可追溯到 checkpoint 和每步非零输出 |
| 只看到工作区，无法确认十物体是否同时存在 | 相机过近 | 视频开头/结尾加入上游、供料架、下游广角 | 可视化镜头也是验收证据的一部分 |
| 物体进入移动碗却被判不稳定 | 使用球的世界速度 | 改用球相对碗速度 | 移动容器的稳定性必须在容器坐标/相对运动中定义 |
| 想靠放宽距离或提前释放解决追不上 | 根因是节拍和跟踪窗口，不是验收过严 | 保持严格判据，修正间距与场景几何 | 不允许用弱判据掩盖吞吐不足 |
| 把 `10/10` 理解成可以无限缩短间距 | 当前是单机械臂串行服务，间距已覆盖完整周期 | 明确当前吞吐上限，压缩节拍需另做流水/多臂/抛放研究 | 成功率与吞吐量是两个指标 |
| 把 MuJoCo 成功等同于真机成功 | 尚无相机误差、通信延迟、安全停机和标定漂移 | 高驱动服务器视觉采集后再做视觉与真机验收 | sim-to-sim 是中间门禁，不是 sim-to-real 终点 |

## 6. 可复现入口

以下四条命令都从仓库根目录执行。MuJoCo 评估与视频显式屏蔽 CUDA；不会占用有散热故障的 GPU0，也不会修改现有 IsaacLab。

任务 A 严格评估：

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  mujoco_sim/evaluate_jaka_ten_sphere_queue.py
```

任务 A CPU 视频：

```bash
CUDA_VISIBLE_DEVICES=-1 LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw \
  xvfb-run -a .venv/bin/python \
  mujoco_sim/render_jaka_ten_sphere_queue.py
```

任务 B 严格评估：

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  mujoco_sim/evaluate_jaka_ten_static_spheres_moving_bowls.py
```

任务 B CPU 视频：

```bash
CUDA_VISIBLE_DEVICES=-1 LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw \
  xvfb-run -a .venv/bin/python \
  mujoco_sim/render_jaka_ten_static_spheres_moving_bowls.py
```

核心代码：

```text
mujoco_sim/jaka_scene.py
  build_jaka_queued_sphere_scene_xml
  build_jaka_static_sphere_moving_bowl_queue_scene_xml

mujoco_sim/jaka_moving_sphere.py
  队列对象索引、配对切换、全局时间载体推进、每轮技能历史重置

mujoco_sim/evaluate_jaka_ten_sphere_queue.py
mujoco_sim/render_jaka_ten_sphere_queue.py
mujoco_sim/evaluate_jaka_ten_static_spheres_moving_bowls.py
mujoco_sim/render_jaka_ten_static_spheres_moving_bowls.py

tests/unit/test_mujoco_moving_sphere.py
```

最近一次完整单元测试基线为 `305 passed`。如果以后更改队列数量、间距、传送带范围、物体绑定或状态机，应重新运行相关单元测试和两项严格评估，不能只重新渲染视频。

## 7. 后续迁移到视觉和真机时的明确边界

当前两个十物体任务验证的是：在精确状态可得的 MuJoCo 环境中，同一冻结混合策略可连续调度十次，并能承受多个物体同时存在和持续运动。尚未证明：

- 双 RGB-D 能可靠识别十个同类球/碗并保持跨帧 ID；
- 视觉系统能给出策略所需的球、碗和机器人相对几何状态；
- 遮挡、反光、深度空洞和标定误差下仍能保持 `10/10`；
- 真机通信延迟和传送带速度波动下仍满足当前节拍；
- 漏抓、占用碗、物体粘连、目标越界或急停后的队列恢复；
- 当前仿真 `10/10` 在更多随机种子和真实扰动下仍成立。

高驱动服务器阶段应为两个任务使用同一套 ZU5+AG95 场景和腕部/外部 RGB-D 相机，采集图像、机器人状态、冻结教师动作、奖励与终止标志。十物体场景还需要增加**目标 ID 与调度标签**，否则视觉学生即使能检测所有目标，也可能在相邻帧切换到错误球或错误碗。

真机端不需要把传送带或物体速度直接输入 actor；仍可依靠连续位姿观测闭环更新。但系统调度层应读取传送带编码器或稳定估计节拍，用于判断哪个目标能进入服务窗口、是否跳过来不及处理的目标。该信息属于队列调度和安全层，不必改变已经冻结的无速度 actor 输入合同。

## 8. 冻结结论

截至 2026-10-26，可以冻结以下工程结论：

1. 两个任务都已从单目标扩展到十目标连续物理队列，并通过正式 checkpoint 闭环 `10/10`；
2. 十目标能力来自“单回合混合策略 + 队列调度 + 足够的空间/时间间距”，不是新训练出的多目标端到端策略；
3. 任务 A 的正式工作点为 `0.08 m/s`、`0.71 m` 间距；任务 B 为 `0.10 m/s`、`0.75 m` 间距；
4. `0.63 m` 失败、逐个刷新版本和零残差版本均保留为过程证据，但不得替代正式 checkpoint 结论；
5. 下一阶段是高驱动服务器上的双相机数据采集与视觉学生，而不是在当前十物体 MuJoCo demo 上继续盲目延长 PPO；
6. 本机所有 Isaac 工作只能使用物理 GPU1；MuJoCo 视频继续使用 CPU，禁止使用 GPU0。
