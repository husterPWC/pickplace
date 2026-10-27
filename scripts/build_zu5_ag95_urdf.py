#!/usr/bin/env python3
"""Build the directly mounted ZU5 + AG95 URDF from the two validated source URDFs."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZU5 = ROOT / "assets/models/manipulators/Zu5/zu5.urdf"
AG95 = ROOT / "assets/models/grippers/Ag95Gripper/ag95.urdf"
OUTPUT = ROOT / "assets/models/robots/Zu5Ag95/zu5_ag95.urdf"


def _append_with_mesh_prefix(dst: ET.Element, src: ET.Element, prefix: str) -> None:
    for child in src:
        item = copy.deepcopy(child)
        for mesh in item.findall(".//mesh"):
            mesh.set("filename", f"{prefix}/{mesh.get('filename')}")
        dst.append(item)


def main() -> None:
    zu5 = ET.parse(ZU5).getroot()
    ag95 = ET.parse(AG95).getroot()
    robot = ET.Element("robot", {"name": "jaka_zu5_ag95"})

    _append_with_mesh_prefix(robot, zu5, "../../manipulators/Zu5")

    mount = ET.SubElement(robot, "joint", {"name": "ag95_mount_joint", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "tool0"})
    ET.SubElement(mount, "child", {"link": "ag95_base_link"})
    # The real setup has only a negligible thin flange: direct frame-to-frame mount.
    ET.SubElement(mount, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    _append_with_mesh_prefix(robot, ag95, "../../grippers/Ag95Gripper")

    ET.indent(robot, space="  ")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
