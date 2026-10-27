"""Isaac Sim application launcher. Call before importing torch in every simulator process."""

from __future__ import annotations

import os
import socket
import sys


def launch_app(headless: bool = True, enable_cameras: bool = False, device: str | None = None):
    """Start Isaac Sim through Isaac Lab's ``AppLauncher`` and return the ``SimulationApp``.

    Raises:
        RuntimeError: if ``torch`` was imported before the app was launched (Isaac Sim requirement).
    """
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch was imported before launching Isaac Sim. Call pickplace.app.launch_app() at the very top "
            "of the entry point, before importing torch, torchrl, tensordict or pickplace.envs."
        )

    # This workstation's physical GPU0 has a cooling fault.  Expose only the
    # selected physical GPU before Isaac Sim or CUDA is imported; inside the
    # process it is intentionally addressed as logical ``cuda:0``.  Refuse an
    # explicit GPU0 request on this host as a fail-safe, while keeping physical
    # GPU selection configurable on the separate multi-GPU server.
    physical_gpu = os.environ.get("PICKPLACE_PHYSICAL_GPU", "1").strip()
    selected_devices = {item.strip() for item in physical_gpu.split(",") if item.strip()}
    if socket.gethostname() == "robot" and "0" in selected_devices:
        raise RuntimeError(
            "Refusing to launch on physical GPU0: host 'robot' GPU0 has a cooling fault. "
            "Use PICKPLACE_PHYSICAL_GPU=1."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu

    from isaaclab.app import AppLauncher

    kwargs: dict = {"headless": headless, "enable_cameras": enable_cameras}
    if device is not None:
        kwargs["device"] = device
    return AppLauncher(**kwargs).app
