"""Spherical food item whose episode reset is owned by the conveyor task."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from pickplace.food.rigid import RigidFoodCfg


@configclass
class ConveyorSphereCfg(RigidFoodCfg):
    """The proven 40 mm sphere without the legacy supply-bowl reset.

    ``MovingCubeStaticBowlEnvCfg`` already resets its free food rigid body on
    the moving pallet.  Keeping ``RigidFoodCfg.reset_food`` as well would move
    the sphere back into the legacy supply bowl during every episode reset.
    All physical sphere parameters otherwise remain identical to the original
    task so this is a clean conveyor-feasibility test.
    """

    name: str = "conveyor_sphere"

    def __post_init__(self):
        super().__post_init__()
        self.events.pop("reset_food", None)
