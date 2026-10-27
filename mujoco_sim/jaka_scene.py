"""Build the clean ZU5+AG95 conveyor scene used by MuJoCo sim-to-sim."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "robot.xml"
ZU5_DIR = ROOT / "assets" / "models" / "manipulators" / "Zu5" / "meshes"
AG95_DIR = ROOT / "assets" / "models" / "grippers" / "Ag95Gripper" / "meshes"


def _element(text: str) -> ET.Element:
    return ET.fromstring(text.strip())


def build_jaka_scene_xml() -> str:
    """Return self-contained MJCF with the uploaded closed-linkage AG95."""
    source = ET.parse(SOURCE).getroot()
    root = ET.Element("mujoco", {"model": "jaka_zu5_ag95_moving_sphere_sim2sim"})
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "autolimits": "true", "balanceinertia": "true"},
    )
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.002",
            "integrator": "implicitfast",
            "gravity": "0 0 -9.81",
            "cone": "elliptic",
            "iterations": "80",
            "noslip_iterations": "4",
            "impratio": "10",
        },
    )
    ET.SubElement(root, "size", {"njmax": "5000", "nconmax": "1000"})
    root.append(
        _element(
            """
            <visual>
              <global azimuth="128" elevation="-24" offwidth="1280" offheight="720"/>
              <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.25 0.25 0.25"/>
              <rgba haze="0.15 0.20 0.25 1"/>
            </visual>
            """
        )
    )

    source_default = source.find("default")
    if source_default is None:
        raise RuntimeError(f"missing AG95 defaults in {SOURCE}")
    root.append(copy.deepcopy(source_default))

    asset = ET.SubElement(root, "asset")
    for material_name in ("grey", "black"):
        material = source.find(f"asset/material[@name='{material_name}']")
        if material is None:
            raise RuntimeError(f"missing material {material_name} in {SOURCE}")
        asset.append(copy.deepcopy(material))
    for mesh in source.findall("asset/mesh"):
        name = mesh.get("name", "")
        if name.startswith("0_Link_"):
            directory = ZU5_DIR
        elif name in {
            "0_base_link", "0_gripper_body", "0_outer_knuckle", "0_inner_knuckle",
            "0_finger", "0_finger_pad_0", "0_finger_pad_1",
        }:
            directory = AG95_DIR
        else:
            continue
        cloned = copy.deepcopy(mesh)
        cloned.set("file", str(directory / Path(mesh.get("file", "")).name))
        asset.append(cloned)
    for name, rgba in {
        "table": "0.22 0.23 0.25 1",
        "belt": "0.055 0.06 0.07 1",
        "pallet": "0.20 0.22 0.24 1",
        "bowl": "0.78 0.76 0.66 1",
        "food": "0.08 0.65 0.18 1",
    }.items():
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})

    world = ET.SubElement(root, "worldbody")
    world.append(_element('<light name="key" pos="0.3 -0.4 2.0" dir="0.1 0.2 -1" diffuse="0.85 0.85 0.85"/>'))
    world.append(_element('<light name="fill" pos="1.3 0.8 1.2" dir="-0.7 -0.3 -0.7" diffuse="0.45 0.50 0.58"/>'))
    world.append(_element('<geom name="floor" type="plane" pos="0 0 -1.05" size="4 4 0.1" rgba="0.10 0.12 0.14 1"/>'))
    world.append(_element('<geom name="table" type="box" pos="0.45 0 -0.40" size="0.75 0.55 0.40" material="table"/>'))
    world.append(_element('<geom name="belt_visual" type="box" pos="0.45 0.30 0.016" size="0.55 0.15 0.004" material="belt" contype="0" conaffinity="0"/>'))

    robot = source.find("worldbody/body[@name='0_mount_base_link']")
    if robot is None:
        raise RuntimeError(f"missing combined ZU5+AG95 body in {SOURCE}")
    robot = copy.deepcopy(robot)
    robot.set("pos", "0 0 0")
    link0 = robot.find("body[@name='0_Link_0']")
    if link0 is None:
        raise RuntimeError("combined asset is missing 0_Link_0")
    link0.set("pos", "0 0 0")
    for body in robot.iter("body"):
        body.set("gravcomp", "1")
    for joint in robot.iter("joint"):
        name = joint.get("name", "")
        if name.startswith("0_joint_"):
            joint.set("armature", "0.1")
            joint.set("damping", "0.1")
        elif name.endswith(("knuckle_joint", "finger_joint")):
            joint.set("range", "0 0.943")
            joint.set("limited", "true")
            joint.set("armature", "0.01")
            joint.set("damping", "0.1")
            joint.set("frictionloss", "0.01")
    # Arm mesh hulls are not needed for this manipulation task and create
    # false self/belt contacts.  AG95 keeps its physical pad/link collisions.
    for body in robot.iter("body"):
        if body.get("name", "").startswith("0_Link_"):
            for geom in body.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    tcp_body = robot.find(".//body[@name='0_eef']")
    tcp = robot.find(".//site[@name='0_grip_site']")
    if tcp_body is None or tcp is None:
        raise RuntimeError("combined asset is missing 0_grip_site")
    # The uploaded standalone gripper used an arbitrary 0.2 m display frame.
    # Replace it with the pad-midpoint TCP measured for the Isaac asset.  The
    # local quaternion maps the uploaded Link_6/tool0 frames into the canonical
    # top-down policy frame (x=finger, y=closing, z=approach-down).
    tcp_body.set("pos", "0 0 0")
    tcp.set("pos", "0 0 -0.16756")
    tcp.set("quat", "0 0.70710678 0.70710678 0")
    tcp.set("rgba", "1 0.35 0.05 0")
    world.append(robot)

    world.append(
        _element(
            """
            <body name="pallet" pos="0.20 0.30 0.02">
              <joint name="pallet_slide" type="slide" axis="1 0 0" range="-0.12 1.0" damping="100"/>
              <inertial pos="0 0 0.005" mass="2" diaginertia="0.012 0.010 0.018"/>
              <geom name="pallet_geom" type="box" pos="0 0 0.005" size="0.11 0.13 0.005"
                material="pallet" friction="1.0 0.02 0.001"/>
            </body>
            """
        )
    )
    bowl = ET.Element("body", {"name": "destination_bowl", "pos": "0.45 -0.10 0"})
    bowl.append(_element('<geom name="bowl_base" type="box" pos="0 0 0.003" size="0.076 0.076 0.003" material="bowl"/>'))
    for index in range(12):
        import math

        angle = 2.0 * math.pi * index / 12
        bowl.append(
            ET.Element(
                "geom",
                {
                    "type": "box",
                    "pos": f"{0.073 * math.cos(angle):.8f} {0.073 * math.sin(angle):.8f} 0.031",
                    "euler": f"0 0 {angle:.8f}",
                    "size": "0.003 0.0226 0.025",
                    "material": "bowl",
                },
            )
        )
    world.append(bowl)
    world.append(
        _element(
            """
            <body name="food" pos="0.20 0.30 0.052">
              <freejoint name="food_free"/>
              <inertial pos="0 0 0" mass="0.03" diaginertia="0.0000048 0.0000048 0.0000048"/>
              <geom name="food_geom" type="sphere" size="0.02" material="food"
                friction="1.5 0.01 0.0005" solimp="0.95 0.99 0.001" solref="0.005 1"/>
            </body>
            """
        )
    )

    for section_name in ("contact", "equality", "tendon"):
        section = source.find(section_name)
        if section is not None:
            root.append(copy.deepcopy(section))

    actuator = ET.SubElement(root, "actuator")
    limits = [(-6.28, 6.28), (-1.48, 4.62), (-3.05, 3.05), (-1.48, 4.62), (-6.28, 6.28), (-6.28, 6.28)]
    forces = [5000, 5000, 5000, 5000, 5000, 5000]
    for index, ((low, high), force) in enumerate(zip(limits, forces), start=1):
        ET.SubElement(
            actuator,
            "general",
            {
                "name": f"arm{index}",
                "joint": f"0_joint_{index}",
                "dyntype": "none",
                "biastype": "affine",
                "ctrllimited": "true",
                "ctrlrange": f"{low} {high}",
                "forcerange": f"-{force} {force}",
                "gainprm": "10000",
                "biasprm": "0 -10000 -500",
            },
        )
    ET.SubElement(
        actuator,
        "position",
        {
            "name": "fingers_actuator",
            "tendon": "0_split",
            "forcerange": "-200 200",
            "ctrlrange": "0 0.943",
            "ctrllimited": "true",
            "kp": "200",
            "kv": "20",
        },
    )
    ET.SubElement(
        actuator,
        "position",
        {
            "name": "pallet_drive",
            "joint": "pallet_slide",
            "kp": "10000",
            "kv": "10000",
            "ctrlrange": "-0.12 1.0",
        },
    )
    return ET.tostring(root, encoding="unicode")


def build_jaka_multi_sphere_scene_xml(num_spheres: int) -> str:
    """Return the frozen JAKA scene with several independently physical balls.

    Sphere zero deliberately keeps the historical ``food`` names so the
    single-ball bridge remains byte-for-byte compatible.  Additional spheres
    are real free bodies; the sequence scheduler enables them one at a time,
    while already placed spheres remain in the destination bowl and continue
    participating in contact dynamics.
    """
    if not 1 <= int(num_spheres) <= 32:
        raise ValueError("num_spheres must be between 1 and 32")
    if int(num_spheres) == 1:
        return build_jaka_scene_xml()

    root = ET.fromstring(build_jaka_scene_xml())
    root.set("model", "jaka_zu5_ag95_multi_sphere_sim2sim")
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("generated JAKA scene is missing worldbody")
    original = world.find("body[@name='food']")
    if original is None:
        raise RuntimeError("generated JAKA scene is missing food")
    insert_at = list(world).index(original) + 1
    for index in range(1, int(num_spheres)):
        sphere = copy.deepcopy(original)
        sphere.set("name", f"food_{index}")
        sphere.set("pos", f"{-2.0 - 0.1 * index:.3f} -2.0 -1.0")
        joint = sphere.find("freejoint")
        geom = sphere.find("geom")
        if joint is None or geom is None:
            raise RuntimeError("food template must contain one freejoint and geom")
        joint.set("name", f"food_free_{index}")
        geom.set("name", f"food_geom_{index}")
        world.insert(insert_at, sphere)
        insert_at += 1
    return ET.tostring(root, encoding="unicode")


def build_jaka_queued_sphere_scene_xml(
    num_spheres: int,
    spacing: float,
) -> str:
    """Return a long conveyor with all balls and carriers present at reset."""
    if not 1 <= int(num_spheres) <= 32:
        raise ValueError("num_spheres must be between 1 and 32")
    if float(spacing) <= 0.0:
        raise ValueError("spacing must be positive")
    root = ET.fromstring(build_jaka_multi_sphere_scene_xml(num_spheres))
    root.set("model", "jaka_zu5_ag95_queued_sphere_sim2sim")
    world = root.find("worldbody")
    actuator = root.find("actuator")
    if world is None or actuator is None:
        raise RuntimeError("generated JAKA scene is missing worldbody or actuator")

    pallet = world.find("body[@name='pallet']")
    drive = actuator.find("position[@name='pallet_drive']")
    if pallet is None or drive is None:
        raise RuntimeError("generated JAKA scene is missing pallet or pallet_drive")
    insert_at = list(world).index(pallet) + 1
    for index in range(1, int(num_spheres)):
        carrier = copy.deepcopy(pallet)
        carrier.set("name", f"pallet_{index}")
        joint = carrier.find("joint")
        geom = carrier.find("geom")
        if joint is None or geom is None:
            raise RuntimeError("pallet template must contain one joint and geom")
        joint.set("name", f"pallet_slide_{index}")
        geom.set("name", f"pallet_geom_{index}")
        world.insert(insert_at, carrier)
        insert_at += 1

        carrier_drive = copy.deepcopy(drive)
        carrier_drive.set("name", f"pallet_drive_{index}")
        carrier_drive.set("joint", f"pallet_slide_{index}")
        actuator.append(carrier_drive)

    upstream = 0.20 - float(spacing) * (int(num_spheres) - 1) - 0.35
    downstream = 1.00
    belt_visual = world.find("geom[@name='belt_visual']")
    table = world.find("geom[@name='table']")
    if belt_visual is None or table is None:
        raise RuntimeError("generated JAKA scene is missing table or belt visual")
    belt_visual.set("pos", f"{0.5 * (upstream + downstream):.8f} 0.30 0.016")
    belt_visual.set("size", f"{0.5 * (downstream - upstream):.8f} 0.15 0.004")
    table_upstream = upstream - 0.20
    table_downstream = 1.20
    table.set("pos", f"{0.5 * (table_upstream + table_downstream):.8f} 0 -0.40")
    table.set("size", f"{0.5 * (table_downstream - table_upstream):.8f} 0.55 0.40")

    lower = upstream - 0.20
    upper = 1.00
    for joint in world.findall("body/joint"):
        if joint.get("name", "").startswith("pallet_slide"):
            joint.set("range", f"{lower:.8f} {upper:.8f}")
    for position in actuator.findall("position"):
        if position.get("name", "").startswith("pallet_drive"):
            position.set("ctrlrange", f"{lower:.8f} {upper:.8f}")
    return ET.tostring(root, encoding="unicode")


def build_jaka_static_sphere_moving_bowl_scene_xml() -> str:
    """Return the independent JAKA scene for static-sphere -> moving-bowl.

    The proven moving-sphere scene is left unchanged.  Its fixed destination
    bowl becomes the off-belt supply bowl at a ZU5-reachable pose, while a
    new free destination bowl rides on the moving pallet exactly like the
    original Franka task.
    """
    root = ET.fromstring(build_jaka_scene_xml())
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("generated JAKA scene is missing worldbody")

    supply = world.find("body[@name='destination_bowl']")
    if supply is None:
        raise RuntimeError("generated JAKA scene is missing destination_bowl")
    supply.set("name", "supply_bowl")
    # Keep the compiled reference pose identical to the validated base scene.
    # The task environment relocates this static body through ``model.body_pos``
    # before reset, matching the already verified JAKA grasp probe exactly.
    supply.set("pos", "0.45 -0.10 0")
    # Replace the deep destination bowl by the original task's wide, shallow
    # supply tray.  AG95's long linkage fingers otherwise collide with the
    # 50 mm destination walls before reaching the sphere.
    for geom in list(supply.findall("geom")):
        supply.remove(geom)
    supply.append(
        _element(
            '<geom name="supply_bowl_base" type="box" pos="0 0 0.003" '
            'size="0.116 0.116 0.003" material="bowl"/>'
        )
    )
    import math

    for index in range(12):
        angle = 2.0 * math.pi * index / 12
        supply.append(
            ET.Element(
                "geom",
                {
                    "type": "box",
                    "pos": (
                        f"{0.113 * math.cos(angle):.8f} "
                        f"{0.113 * math.sin(angle):.8f} 0.0135"
                    ),
                    "euler": f"0 0 {angle:.8f}",
                    "size": "0.003 0.0333 0.0075",
                    "material": "bowl",
                },
            )
        )

    # The 0.10 m/s curriculum needs enough approach time for the independent
    # static-sphere grasp.  Extend only this scene's conveyor entry; the
    # frozen moving-sphere scene retains its original dimensions/ranges.
    belt_visual = world.find("geom[@name='belt_visual']")
    if belt_visual is not None:
        belt_visual.set("pos", "0.35 0.30 0.016")
        belt_visual.set("size", "0.75 0.15 0.004")
    table = world.find("geom[@name='table']")
    if table is not None:
        table.set("size", "0.75 0.70 0.40")
    pallet_joint = world.find("body[@name='pallet']/joint[@name='pallet_slide']")
    if pallet_joint is not None:
        pallet_joint.set("range", "-0.50 1.0")
    pallet_actuator = root.find("actuator/position[@name='pallet_drive']")
    if pallet_actuator is not None:
        pallet_actuator.set("ctrlrange", "-0.50 1.0")

    moving = ET.Element("body", {"name": "moving_bowl", "pos": "0.20 0.30 0.032"})
    moving.append(ET.Element("freejoint", {"name": "moving_bowl_free"}))
    moving.append(
        ET.Element(
            "inertial",
            {
                "pos": "0 0 0.025",
                "mass": "0.2",
                "diaginertia": "0.0012 0.0012 0.0018",
            },
        )
    )
    moving.append(
        _element(
            '<geom name="moving_bowl_base" type="box" pos="0 0 0.003" '
            'size="0.076 0.076 0.003" material="bowl" friction="1.0 0.02 0.001"/>'
        )
    )
    for index in range(12):
        angle = 2.0 * math.pi * index / 12
        moving.append(
            ET.Element(
                "geom",
                {
                    "type": "box",
                    "pos": (
                        f"{0.073 * math.cos(angle):.8f} "
                        f"{0.073 * math.sin(angle):.8f} 0.031"
                    ),
                    "euler": f"0 0 {angle:.8f}",
                    "size": "0.003 0.0226 0.025",
                    "material": "bowl",
                },
            )
        )
    food = world.find("body[@name='food']")
    if food is None:
        raise RuntimeError("generated JAKA scene is missing food")
    world.insert(list(world).index(food), moving)
    return ET.tostring(root, encoding="unicode")


def build_jaka_static_sphere_moving_bowl_queue_scene_xml(
    num_items: int,
    spacing: float,
) -> str:
    """Return a long conveyor with a static ball rack and moving bowl queue."""
    if not 1 <= int(num_items) <= 32:
        raise ValueError("num_items must be between 1 and 32")
    if float(spacing) <= 0.0:
        raise ValueError("spacing must be positive")
    root = ET.fromstring(build_jaka_static_sphere_moving_bowl_scene_xml())
    root.set("model", "jaka_zu5_ag95_static_spheres_moving_bowls_queue")
    world = root.find("worldbody")
    actuator = root.find("actuator")
    if world is None or actuator is None:
        raise RuntimeError("generated JAKA scene is missing worldbody or actuator")

    food = world.find("body[@name='food']")
    bowl = world.find("body[@name='moving_bowl']")
    pallet = world.find("body[@name='pallet']")
    drive = actuator.find("position[@name='pallet_drive']")
    supply = world.find("body[@name='supply_bowl']")
    if food is None or bowl is None or pallet is None or drive is None or supply is None:
        raise RuntimeError("static-to-moving scene is missing a queue template")

    food_insert = list(world).index(food) + 1
    bowl_insert = list(world).index(bowl) + 1
    pallet_insert = list(world).index(pallet) + 1
    for index in range(1, int(num_items)):
        sphere = copy.deepcopy(food)
        sphere.set("name", f"food_{index}")
        sphere.set("pos", f"{-2.0 - 0.1 * index:.3f} -2.0 -1.0")
        sphere_joint = sphere.find("freejoint")
        sphere_geom = sphere.find("geom")
        if sphere_joint is None or sphere_geom is None:
            raise RuntimeError("food template must contain one freejoint and geom")
        sphere_joint.set("name", f"food_free_{index}")
        sphere_geom.set("name", f"food_geom_{index}")
        world.insert(food_insert, sphere)
        food_insert += 1

        moving_bowl = copy.deepcopy(bowl)
        moving_bowl.set("name", f"moving_bowl_{index}")
        bowl_joint = moving_bowl.find("freejoint")
        bowl_base = moving_bowl.find("geom[@name='moving_bowl_base']")
        if bowl_joint is None or bowl_base is None:
            raise RuntimeError("moving bowl template is missing its freejoint or base")
        bowl_joint.set("name", f"moving_bowl_free_{index}")
        bowl_base.set("name", f"moving_bowl_base_{index}")
        world.insert(bowl_insert, moving_bowl)
        bowl_insert += 1

        carrier = copy.deepcopy(pallet)
        carrier.set("name", f"pallet_{index}")
        carrier_joint = carrier.find("joint")
        carrier_geom = carrier.find("geom")
        if carrier_joint is None or carrier_geom is None:
            raise RuntimeError("pallet template must contain one joint and geom")
        carrier_joint.set("name", f"pallet_slide_{index}")
        carrier_geom.set("name", f"pallet_geom_{index}")
        world.insert(pallet_insert, carrier)
        pallet_insert += 1

        carrier_drive = copy.deepcopy(drive)
        carrier_drive.set("name", f"pallet_drive_{index}")
        carrier_drive.set("joint", f"pallet_slide_{index}")
        actuator.append(carrier_drive)

    # Ten static spheres use a shallow 2 x 5 rack.  Low walls preserve real
    # support while leaving AG95's linkage clear during top-down pickup.
    for geom in list(supply.findall("geom")):
        supply.remove(geom)
    supply.append(
        _element(
            '<geom name="supply_bowl_base" type="box" pos="0 0 0.003" '
            'size="0.170 0.105 0.003" material="bowl"/>'
        )
    )
    supply.append(_element('<geom type="box" pos="0.167 0 0.0135" size="0.003 0.105 0.0075" material="bowl"/>'))
    supply.append(_element('<geom type="box" pos="-0.167 0 0.0135" size="0.003 0.105 0.0075" material="bowl"/>'))
    supply.append(_element('<geom type="box" pos="0 0.102 0.0135" size="0.170 0.003 0.0075" material="bowl"/>'))
    supply.append(_element('<geom type="box" pos="0 -0.102 0.0135" size="0.170 0.003 0.0075" material="bowl"/>'))

    upstream = 0.08 - float(spacing) * (int(num_items) - 1) - 0.35
    # Keep the first filled bowl physically supported until the tenth place.
    downstream = 0.08 + float(spacing) * int(num_items) + 0.55
    belt_visual = world.find("geom[@name='belt_visual']")
    table = world.find("geom[@name='table']")
    if belt_visual is None or table is None:
        raise RuntimeError("generated JAKA scene is missing table or belt visual")
    belt_visual.set("pos", f"{0.5 * (upstream + downstream):.8f} 0.30 0.016")
    belt_visual.set("size", f"{0.5 * (downstream - upstream):.8f} 0.15 0.004")
    table_upstream = upstream - 0.20
    table_downstream = downstream + 0.20
    table.set("pos", f"{0.5 * (table_upstream + table_downstream):.8f} 0 -0.40")
    table.set("size", f"{0.5 * (table_downstream - table_upstream):.8f} 0.70 0.40")

    lower = upstream - 0.20 - 0.20
    upper = downstream - 0.20
    for joint in world.findall("body/joint"):
        if joint.get("name", "").startswith("pallet_slide"):
            joint.set("range", f"{lower:.8f} {upper:.8f}")
    for position in actuator.findall("position"):
        if position.get("name", "").startswith("pallet_drive"):
            position.set("ctrlrange", f"{lower:.8f} {upper:.8f}")
    return ET.tostring(root, encoding="unicode")
