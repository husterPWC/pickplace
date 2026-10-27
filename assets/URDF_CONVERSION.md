# ZU5 and AG95 conversion notes

## Generated assets

- `models/manipulators/Zu5/zu5.urdf`
- `models/manipulators/Zu5/usd/zu5/zu5.usda`
- `models/grippers/Ag95Gripper/ag95.urdf`
- `models/grippers/Ag95Gripper/usd/ag95/ag95.usda`

The URDF files use paths relative to their own directories, so they can be
imported without a ROS package path. The USD directories were generated with
the repository-local Isaac Lab/Isaac Sim installation and a fixed base.

## ZU5

The uploaded `Zu5.xml` is the primary source for kinematics, joint limits,
effort limits, and link inertials. The joint velocity limit (`3.14 rad/s`) and
base-link inertial were cross-checked against JAKA's official ROS 2 ZU5 URDF:

<https://github.com/JAKARobotics/jaka_ros2>

The `tool0` frame preserves the uploaded MJCF attachment transform: it is at
the Link_6 origin with a 180-degree rotation about Z.

## AG95

The uploaded AG95 MJCF describes a closed four-bar linkage using MuJoCo
equality constraints. A URDF is necessarily a tree and cannot encode that
closed loop. The generated model therefore uses the usual approximation:

- `left_outer_knuckle_joint` is the one commanded joint (`0..0.943 rad`).
- The other five revolute joints mimic it with multiplier `1`.
- Inertial properties were calculated by compiling the uploaded collision
  geometry and its `2700 kg/m^3` density with MuJoCo.
- Millimetre meshes use scale `0.001`; the two pad meshes are already in
  metres and use scale `1`.

Isaac Sim's importer retained all five mimic relationships as
`NewtonMimicAPI` relationships in the generated USD.

## Combining the arm and gripper

The models remain separate intentionally. A combined articulation needs the
measured transform from the ZU5 flange/adapter (`tool0`) to
`ag95_base_link`. Do not guess this transform for training: an incorrect
rotation or tool offset changes inverse kinematics, collision geometry, and
sim-to-real calibration. Once the adapter transform is known, connect the two
roots with a fixed joint and import the combined URDF as one articulation.

The current collision meshes are suitable for initial import validation.
Before large parallel training, profile them and use convex decompositions or
simplified primitives if contact simulation becomes the bottleneck.
