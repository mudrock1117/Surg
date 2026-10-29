#!/usr/bin/env python3
"""Preview the two eye-section STL parts and ILM membrane in PyBullet.

This is a visual-only staging tool. STL files do not carry the SolidWorks
assembly-instance transforms, so the transforms must be supplied explicitly.
Do not treat the default zero transforms as a calibrated assembly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pybullet as p


def _vec3(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a list of 3 numbers")
    result = [float(v) for v in value]
    if not all(abs(v) < float("inf") for v in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "units": "mm",
            "objects": {
                "eye_part_1": {"position_m": [0, 0, 0], "euler_deg": [0, 0, 0]},
                "eye_part_2": {"position_m": [0, 0, 0], "euler_deg": [0, 0, 0]},
                "membrane": {"position_m": [0, 0, 0], "euler_deg": [0, 0, 0]},
            },
        }
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("objects"), dict):
        raise ValueError("Config must contain an 'objects' object")
    return config


def _resolve_mesh(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STL not found: {path}")
    if path.suffix.lower() != ".stl":
        raise ValueError(f"Expected an STL file: {path}")
    return path


def _add_mesh(
    label: str,
    mesh_path: Path,
    object_config: dict[str, Any],
    scale: float,
    rgba: list[float],
) -> int:
    position = _vec3(object_config.get("position_m", [0, 0, 0]), f"{label}.position_m")
    euler_deg = _vec3(object_config.get("euler_deg", [0, 0, 0]), f"{label}.euler_deg")
    orientation = p.getQuaternionFromEuler([v * 3.141592653589793 / 180.0 for v in euler_deg])

    visual_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=str(mesh_path),
        meshScale=[scale, scale, scale],
        rgbaColor=rgba,
    )
    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        baseCollisionShapeIndex=-1,  # visual-only: avoid unreliable raw-mesh collisions
        basePosition=position,
        baseOrientation=orientation,
    )
    print(
        f"[loaded] {label}: {mesh_path}\n"
        f"         position_m={position}, euler_deg={euler_deg}, scale={scale}"
    )
    return body_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", required=True, help="STL path for Human Eye2<1>")
    parser.add_argument("--part2", required=True, help="STL path for Human Eye2<4>")
    parser.add_argument("--membrane", required=True, help="STL path for membrane<1>")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON file containing each object's position_m and euler_deg",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.001,
        help="Uniform mesh scale (default 0.001; assumes STL coordinates are mm)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    if args.scale <= 0:
        parser.error("--scale must be greater than zero")

    repo_root = Path(__file__).resolve().parents[2]
    meshes = {
        "eye_part_1": _resolve_mesh(args.part1, repo_root),
        "eye_part_2": _resolve_mesh(args.part2, repo_root),
        "membrane": _resolve_mesh(args.membrane, repo_root),
    }
    config_path = args.config
    if config_path is not None and not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    config = _load_config(config_path)
    objects = config["objects"]

    missing = [name for name in meshes if name not in objects]
    if missing:
        raise ValueError(f"Config is missing object entries: {', '.join(missing)}")

    print("\nWARNING: default transforms are placeholders, not SolidWorks assembly transforms.")
    print("If the two eye STL exports share local coordinates, they may overlap exactly.")
    print("Use --config with measured assembly transforms before evaluating alignment.\n")

    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Could not connect to PyBullet GUI")

    try:
        p.resetSimulation()
        p.setGravity(0, 0, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        colors = {
            "eye_part_1": [0.86, 0.86, 0.82, 1.0],
            "eye_part_2": [0.25, 0.55, 0.95, 0.45],
            "membrane": [1.0, 0.22, 0.18, 1.0],
        }
        for name, mesh_path in meshes.items():
            _add_mesh(name, mesh_path, objects[name], args.scale, colors[name])

        # World axes at the world origin (red=X, green=Y, blue=Z).
        axis_len = 0.05
        p.addUserDebugLine([0, 0, 0], [axis_len, 0, 0], [1, 0, 0], 2)
        p.addUserDebugLine([0, 0, 0], [0, axis_len, 0], [0, 1, 0], 2)
        p.addUserDebugLine([0, 0, 0], [0, 0, axis_len], [0, 0, 1], 2)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.12,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0],
        )

        print("PyBullet eye-scene preview is running. Close the GUI to exit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / 60.0)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
