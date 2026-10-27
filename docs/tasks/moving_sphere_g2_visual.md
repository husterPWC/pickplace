# G2：双相机 RGB-D 视觉数据阶段

## 冻结入口

G2 不再继续修改状态 Teacher。唯一冻结入口为：

`artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g1_place_008_1024_no_rtx/checkpoints/ppo_teacher_262144.pt`

该节点在 Isaac 独立确定性评估中达到 221/256 严格完整成功，在 MuJoCo 随机初态纯接触验收中达到
20/20。G2 的目标不是重新训练 G1，而是用它生成视觉监督并训练可真机部署的视觉学生。

## 学生运行时输入

- 固定俯视相机：RGB + 深度；
- 腕部相机：RGB + 深度；
- 机械臂关节位置、夹爪编码器、TCP 位姿、上一动作；
- 标定后的静止目标碗位姿 `belt/bowl_pos`。

明确禁止作为学生输入：物体速度、传送带速度、双指接触、法向力、触觉、电机电流和扭矩。仿真中的
`grasp_guidance` 与 `position_history` 只作为 Teacher 标签保存，可用于几何监督或辅助损失，不能在
部署时喂给学生。G1 输出的动作是环境确定性抓放控制器之前的受限 PPO 残差；数据集用
`teacher/residual_action` 明确记录这一语义，不能把它误当成关节或 TCP 的完整命令。

## 服务器执行顺序

本阶段的真实相机渲染需要高驱动服务器。本机只做无仿真的合同、脚本和数据验收，不在本机启动 RTX
相机，也不修改 `/data/CodePWC/IsaacLab`。服务器上显式选择一张健康 GPU：

```bash
cd /path/to/pickplace
PICKPLACE_PHYSICAL_GPU=<server-gpu> scripts/run_g2_collection_server.sh smoke
```

wrapper 会依次验证冻结权重、采集 4096 帧、检查 RGB/深度/状态/标签/动作语义和 checkpoint 哈希，
并生成 shard 内的 `g2_rgbd_preview.png`。验收标准为：

1. 两路 RGB 均非黑屏、非恒定图；
2. 两路深度为有限的 float16 米制数据，范围 `[0,3] m`，且能看到 3 m 内几何；
3. 学生运行时字段齐全，不含禁用信号；
4. Teacher 标签齐全，`teacher/residual_action` 与实际执行的策略残差一致；
5. manifest 固定 `frame_stack=1`、无速度状态 Teacher、G1 动作语义与来源哈希。

smoke 通过后才运行：

```bash
PICKPLACE_PHYSICAL_GPU=<server-gpu> G2_SEED=0 scripts/run_g2_collection_server.sh pilot
PICKPLACE_PHYSICAL_GPU=<server-gpu> G2_SEED=0 scripts/run_g2_collection_server.sh full
```

正式数据应使用多个 seed 建独立 shard，不覆盖已有 shard。84x84、双 RGB-D、包含 Teacher 几何标签
时，保守估算约 83 GB/百万帧，开始 full 前应先检查服务器磁盘空间和 pilot 的采集吞吐。

## 后续门禁

G2 的后续顺序固定为：视觉数据 smoke → pilot 数据分布检查 → 多 seed 正式采集 → 视觉学生监督/蒸馏
训练 → Isaac 视觉闭环评估 → MuJoCo/真机接口验证。双相机采集与当前抓取位姿逻辑不冲突：相机是
学生的可部署观测，6D 抓取几何是 Teacher-only 监督。不得为了提高离线指标把隐藏几何重新接回学生
actor，否则会产生无法在真机复现的输入依赖。
