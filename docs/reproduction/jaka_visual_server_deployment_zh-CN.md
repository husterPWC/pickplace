# JAKA 双任务高驱动服务器部署与双 RGB-D 采集

> 建立日期：2026-10-27  
> 当前阶段：服务器尚未部署；本文件固定部署顺序、Teacher 来源和烟雾测试门禁。  
> 原则：服务器使用独立的固定版本环境，不修改本机 `/data/CodePWC/IsaacLab`；本机 GPU0 永久禁用。

## 1. 本阶段到底做什么

当前十球/十碗 Demo 没有训练新的十物体网络，而是分别调度两个冻结状态 Teacher。高驱动服务器阶段的目标，是在 Isaac 中给这两个 Teacher 打开腕部与外部 RGB-D 相机，并行生成视觉学生数据：

```text
精确仿真状态 → 冻结状态 Teacher → 7 维残差动作 → JAKA 技能控制器
       ↓              同一个仿真控制步               ↓
双相机 RGB/深度 + 本体状态 + Teacher 标签 + 奖励/终止/阶段
```

第一批数据只做单球/单碗 episode。十物体队列先保留为后续视觉策略的调度验收，避免一开始把目标 ID、遮挡和队列调度混进基础视觉学习问题。

## 2. 两个冻结 Teacher 合约

### 2.1 动态小球 → 静止碗

正式 actor 使用 J0-S1 权重：

```text
artifacts/jaka_zu5_ag95_sphere/teachers/
  jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt

SHA-256:
14b1d8af7ecaa46e4a77c48434ca07d43e7d2f3e73672832af663ece2485ea7b
```

该 checkpoint 自身记录的是抓取/抬升任务，不包含完整放置环境。因此采集时必须单独从 J1 checkpoint 读取完整环境配置：

```text
artifacts/jaka_zu5_ag95_sphere/teachers/
  jaka_zu5_ag95_sphere_j1_place_006_008_1024_no_rtx/checkpoints/ppo_teacher_final.pt

SHA-256:
a945612cfb694adc424b7264635c20a34279e89e08c9aa20dc93c79c47450759
```

采集器明确区分 `checkpoint` 和 `env_checkpoint`。不得直接只传 J0-S1，否则会退回 lift-only 环境；也不得用 J1 actor 替换已验收的 J0-S1 actor。

### 2.2 静止小球 → 0.10 m/s 动态碗

actor 和完整环境都来自 M0：

```text
artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/
  jaka_static_sphere_moving_bowl_m0_final_geometry_smoke16/
  checkpoints/ppo_teacher_final.pt

SHA-256:
d522d818dccdb7a073708b75ccb44fb7bc9b98a5780e10f057b5faba3b01555b
```

运行脚本会在启动 Isaac 前校验这些哈希，错误权重不会进入采集。

## 3. 服务器部署顺序

服务器获得本仓库代码后，先在仓库根目录执行只读预检：

```bash
scripts/check_jaka_visual_server.sh
```

预检检查 Git、Docker、NVIDIA 工具、GPU 列表、Docker 权限和可用空间。两个 100 万帧、84×84 双 RGB-D shard 的保守估计各约 83 GB，加上 Isaac/Docker 缓存和中间文件，工作盘至少保留 250 GiB，建议 350 GiB 以上。

环境优先使用仓库固定 Docker 路线：

```bash
./docker/build.sh
./docker/run.sh python scripts/verify_install.py
```

在共享八卡服务器上必须显式限制可见 GPU。例如 GPU7 空闲时使用：

```bash
FOOD_ROBOT_GPUS=device=7 ./docker/run.sh python scripts/verify_install.py
```

容器内显示的 `cuda:0` 此时映射到物理 GPU7；不要同时再传物理编号 `7` 给容器内程序。后续所有 Isaac 采集命令都沿用同一个 `FOOD_ROBOT_GPUS=device=7` 前缀。

固定基线来自当前仓库：

```text
Isaac Lab: v3.0.0-beta2.patch1
Isaac Sim: 6.0.1
Python:    3.12
TorchRL:   >=0.14,<0.15
TensorDict:>=0.14.2,<0.15
```

不要在服务器上使用“最新版本”替换这些固定版本。先复现固定环境；升级必须建立独立实验目录并重新做 Teacher 回归测试。

## 4. 权重传输

`artifacts/` 被 `.gitignore` 排除，服务器 `git clone` 后不会自动获得三个 checkpoint。必须按第 2 节的相对路径复制，并在服务器执行：

```bash
sha256sum \
  artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt \
  artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j1_place_006_008_1024_no_rtx/checkpoints/ppo_teacher_final.pt \
  artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/jaka_static_sphere_moving_bowl_m0_final_geometry_smoke16/checkpoints/ppo_teacher_final.pt
```

输出必须与第 2 节完全一致。权重只需约 6.5 MB/个；不要复制本机全部历史训练目录。

本机可生成只包含这三个固定文件的传输包：

```bash
scripts/package_jaka_visual_server_weights.sh
```

默认输出位于：

```text
artifacts/server_transfer/jaka_visual_teacher_payload_20261027.tar.gz
```

把压缩包传到服务器仓库根目录后执行 `tar -xzf <压缩包>`，即可恢复原相对路径。包内还包含 `deploy/jaka_visual_teacher_manifest.json`；解压后仍须运行采集脚本的 dry-run，让脚本再次校验文件哈希和 checkpoint 内部合约。

## 5. 必须按 smoke → pilot → full 推进

以下命令在固定项目环境内执行。若使用 Docker，可把脚本作为 `docker/run.sh` 的命令传入。

先打印解析后的合约，不启动 Isaac：

```bash
JAKA_VISUAL_DRY_RUN=1 \
  scripts/run_jaka_visual_collection_server.sh moving_sphere_static_bowl smoke

JAKA_VISUAL_DRY_RUN=1 \
  scripts/run_jaka_visual_collection_server.sh static_sphere_moving_bowl smoke
```

然后分别采集 4096 帧 smoke：

```bash
scripts/run_jaka_visual_collection_server.sh moving_sphere_static_bowl smoke
scripts/run_jaka_visual_collection_server.sh static_sphere_moving_bowl smoke
```

每个 smoke 自动执行 shard 校验并生成 `jaka_rgbd_preview.png`。只有以下条件全部满足才能进入 10 万帧 pilot：

- 腕部和外部 RGB 都不是空白、全黑或固定帧；
- 两路深度为有限的米制数据，范围在 `[0, 3] m`；
- 腕部画面包含两指、球和接近区域，且没有被手腕/法兰完全遮挡；
- 外部画面覆盖传送带、抓取窗口、供料区与目标碗；
- RGB、深度、动作、奖励和终止标志属于同一控制步；
- Teacher 仍保持原任务的严格成功逻辑；
- manifest 中 actor 与环境 checkpoint 哈希正确；
- 网络动作语义为 `teacher_residual_before_environment_jaka_skill_controller`；
- actor 输入不含物体速度、带速、接触、力、触觉、电流或扭矩。

smoke 通过后运行 pilot：

```bash
scripts/run_jaka_visual_collection_server.sh moving_sphere_static_bowl pilot
scripts/run_jaka_visual_collection_server.sh static_sphere_moving_bowl pilot
```

pilot 需要统计视觉开启后的成功率、失败类型、显存、主存、帧率和磁盘写入速度。成功率与无相机 Teacher 明显不一致时停止，不进入 full。确认相机外参与性能后再运行：

```bash
scripts/run_jaka_visual_collection_server.sh moving_sphere_static_bowl full
scripts/run_jaka_visual_collection_server.sh static_sphere_moving_bowl full
```

## 6. 相机当前的已知风险

外部相机已有可配置的眼点和目标点；JAKA 腕部相机挂在 ZU5 `Link_6` 下。但当前腕部外参在 `pickplace/arms/zu5_ag95.py` 中明确是占位安装：

```text
pos=(0.06, 0.0, 0.10)
rot=(0.0, -0.173648, 0.0, 0.984808), convention="ros"
```

所以服务器第一次 smoke 的主要任务不是追求数据量，而是校准视野：夹爪闭合过程中球不能离开画面，夹爪自身不能遮满视野，深度近裁剪面不能切掉手指或球。若修改相机外参，应保存修改前后的 preview 和独立配置，不得用一个未经记录的值覆盖所有历史数据。

## 7. 数据输入边界

视觉 Student 的运行时输入固定为：

- 外部 RGB、腕部 RGB；
- 外部深度、腕部深度；
- 六轴关节相对位置；
- AG95 主关节位置；
- TCP 位置和四元数；
- 上一步动作；
- 碗位置：静止碗可由标定固定，动态碗由外部视觉/标定跟踪获得。

Teacher-only 标签包括 40×20 抓取几何点、最佳抓取位姿差和位置历史。它们可以用于辅助监督，但不能在真机 Student actor 推理时直接读取。策略仍不输入物体速度、传送带速度、接触、力、触觉、电流或扭矩。

## 8. 进入十物体视觉队列的条件

必须先让两个单目标视觉 Student 各自通过独立闭环评估。之后才增加：

1. 十个同类目标的检测与跨帧 ID；
2. 球—碗配对和当前服务目标锁定；
3. 遮挡期间的目标保持；
4. 来不及服务目标的跳过与异常恢复；
5. 最终十物体连续队列验收。

十物体调度层可以使用传送带编码器或节拍估计，但这些信息不作为 actor 的速度输入。
