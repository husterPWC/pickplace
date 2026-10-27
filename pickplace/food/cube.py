"""Rigid cube used by the moving-object/static-bowl task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.sim.schemas import CollisionPropertiesCfg, RigidBodyPropertiesCfg

from pickplace.envs.mdp.observations import asset_quat_w
from pickplace.food.base import FoodSourceCfg


@configclass
class RigidCubeCfg(FoodSourceCfg):
    """A 4 cm cube whose episode reset is owned by the conveyor task."""

    name: str = "rigid_cube"
    size: float = 0.04
    item_radius: float = 0.02
    mass: float = 0.04
    static_friction_range: tuple[float, float] = (0.8, 1.2)
    dynamic_friction_range: tuple[float, float] = (0.7, 1.0)
    restitution_range: tuple[float, float] = (0.0, 0.05)
    mass_scale_range: tuple[float, float] = (0.8, 1.2)

    def __post_init__(self):
        self.item_radius = 0.5 * self.size
        self.asset = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Food",
            spawn=sim_utils.CuboidCfg(
                size=(self.size, self.size, self.size),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),
                collision_props=CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.85),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.19, 0.06), roughness=0.65),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        # Reset is deliberately absent here: the task must place the cube on the
        # current pallet and give it the sampled conveyor velocity atomically.
        self.events = {
            "food_material": EventTerm(
                func=base_mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("food"),
                    "static_friction_range": self.static_friction_range,
                    "dynamic_friction_range": self.dynamic_friction_range,
                    "restitution_range": self.restitution_range,
                    "num_buckets": 32,
                    "make_consistent": True,
                },
            ),
            "food_mass": EventTerm(
                func=base_mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("food"),
                    "mass_distribution_params": self.mass_scale_range,
                    "operation": "scale",
                },
            ),
        }
        self.privileged_obs = {
            "food_quat": ObsTerm(func=asset_quat_w, params={"asset_cfg": SceneEntityCfg("food")}),
        }
