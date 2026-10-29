#!/usr/bin/env python3
"""Diagnose and preview eye-section STL meshes and the ILM membrane in PyBullet.

The preview is visual-only. STL files generally do not preserve SolidWorks
assembly-instance transforms. Use --config to supply measured transforms.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import time
from pathlib import Path
from typing import Any

import pybullet as p


def _vec3(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a list of 3 numbers")
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _load_config(path: Path | None) -> dict[str, Any]:
    defaults = {
        "objects": {
            "eye_part_1": {"position_m": [0, 0, 0], "euler_deg": [0, 0, 0]},
            "eye_part_2": {"position_m": [0, 0, 0], "euler_deg": [0, 0, 0]},
            "membrane": {"position_m": [0, 0, 0], "euler_deg": [0, 0, 0]},
        }
    }
    if path is None:
        return defaults
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


def _read_stl_vertices(path: Path) -> list[tuple[float, float, float]]:
    """Read vertices from binary or ASCII STL without third-party packages."""
    raw = path.read_bytes()
    # Binary STL: 80-byte header + uint32 triangle count + 50 bytes/triangle.
    if len(raw) >= 84:
        triangle_count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + triangle_count * 50 == len(raw):
            vertices: list[tuple[float, float, float]] = []
            offset = 84
            for _ in range(triangle_count):
                # Normal (3 floats), then three vertices (9 floats), then attribute bytes.
                values = struct.unpack_from("<12f", raw, offset)
                vertices.extend((values[3:6], values[6:9], values[9:12]))
                offset += 50
            if vertices:
                return vertices
    # ASCII STL fallback.
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception as exc:
        raise ValueError(f"Cannot decode STL: {path}") from exc
    vertices = [
        (float(x), float(y), float(z))
        for x, y, z in re.findall(
            r"\bvertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)",
            text,
            flags=re.IGNORECASE,
        )
    ]
    if not vertices:
        raise ValueError(f"No triangle vertices found; unsupported or empty STL: {path}")
    return vertices


def _mesh_bounds(
    mesh_path: Path,
    position: list[float],
    euler_deg: list[float],
    scale: float,
) -> tuple[list[float], list[float], list[float], list[float], int]:
    """Return local dimensions, transformed world bounds, and vertex count."""
    vertices = _read_stl_vertices(mesh_path)
    local_min = [min(v[i] for v in vertices) for i in range(3)]
    local_max = [max(v[i] for v in vertices) for i in range(3)]
    local_dims = [local_max[i] - local_min[i] for i in range(3)]

    roll, pitch, yaw = [math.radians(v) for v in euler_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Rotation matrix matching roll-pitch-yaw (XYZ Euler) convention.
    rotation = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    transformed = []
    for vertex in vertices:
        scaled = [component * scale for component in vertex]
        transformed.append([
            sum(rotation[i][j] * scaled[j] for j in range(3)) + position[i]
            for i in range(3)
        ])
    world_min = [min(v[i] for v in transformed) for i in range(3)]
    world_max = [max(v[i] for v in transformed) for i in range(3)]
    return local_dims, world_min, world_max, local_min, len(vertices)


def _add_mesh(
    label: str,
    mesh_path: Path,
    object_config: dict[str, Any],
    scale: float,
    rgba: list[float],
) -> tuple[int, list[float], list[float]]:
    position = _vec3(object_config.get("position_m", [0, 0, 0]), f"{label}.position_m")
    euler_deg = _vec3(object_config.get("euler_deg", [0, 0, 0]), f"{label}.euler_deg")
    euler_rad = [math.radians(v) for v in euler_deg]
    orientation = p.getQuaternionFromEuler(euler_rad)

    visual_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=str(mesh_path),
        meshScale=[scale, scale, scale],
        rgbaColor=rgba,
    )
    if visual_id < 0:
        raise RuntimeError(f"PyBullet failed to create visual mesh for {label}: {mesh_path}")
    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        # Intentionally visual-only: no collision shape is created at this stage.
        baseCollisionShapeIndex=-1,
        basePosition=position,
        baseOrientation=orientation,
    )

    local_dims, world_min, world_max, local_min, vertex_count = _mesh_bounds(
        mesh_path, position, euler_deg, scale
    )
    world_dims = [world_max[i] - world_min[i] for i in range(3)]
    print(f"\n[loaded] {label}")
    print(f"  file               : {mesh_path}")
    print(f"  body ID            : {body_id}")
    print(f"  STL vertices       : {vertex_count}")
    print(f"  STL local min      : {[round(v, 6) for v in local_min]}")
    print(f"  STL size (raw units): {[round(v, 6) for v in local_dims]}")
    print(f"  position m         : {[round(v, 6) for v in position]}")
    print(f"  Euler deg          : {euler_deg}")
    print(f"  scale              : {scale}")
    print(f"  world bounds min m : {[round(v, 6) for v in world_min]}")
    print(f"  world bounds max m : {[round(v, 6) for v in world_max]}")
    print(f"  world size m       : {[round(v, 6) for v in world_dims]}")
    print(f"  world size mm      : {[round(v * 1000, 3) for v in world_dims]}")
    return body_id, world_min, world_max


def _fit_camera(bounds: list[tuple[list[float], list[float]]], fallback_distance: float = 0.12) -> None:
    if not bounds:
        return
    lower = [min(box[0][i] for box in bounds) for i in range(3)]
    upper = [max(box[1][i] for box in bounds) for i in range(3)]
    center = [(lower[i] + upper[i]) / 2.0 for i in range(3)]
    extent = max(upper[i] - lower[i] for i in range(3))
    distance = max(extent * 2.8, 0.01) if extent > 0 else fallback_distance
    print("\n[scene]")
    print(f"  combined bounds min m: {[round(v, 6) for v in lower]}")
    print(f"  combined bounds max m: {[round(v, 6) for v in upper]}")
    print(f"  camera target m      : {[round(v, 6) for v in center]}")
    print(f"  auto camera distance : {distance:.4f} m")
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=center,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", required=True, help="STL for Human Eye2<1>")
    parser.add_argument("--part2", required=True, help="STL for Human Eye2<4>")
    parser.add_argument("--membrane", required=True, help="STL for membrane<1>")
    parser.add_argument("--config", type=Path, help="JSON with each object's position_m and euler_deg")
    parser.add_argument(
        "--show",
        choices=("all", "part1", "part2", "membrane"),
        default="all",
        help="Show all meshes or one mesh at a time (default: all)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.001,
        help="Uniform mesh scale; default 0.001 assumes STL coordinates are millimeters",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    if args.scale <= 0 or not math.isfinite(args.scale):
        parser.error("--scale must be a finite number greater than zero")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    all_meshes = {
        "eye_part_1": _resolve_mesh(args.part1, repo_root),
        "eye_part_2": _resolve_mesh(args.part2, repo_root),
        "membrane": _resolve_mesh(args.membrane, repo_root),
    }
    show_to_name = {"part1": "eye_part_1", "part2": "eye_part_2", "membrane": "membrane"}
    meshes = all_meshes if args.show == "all" else {show_to_name[args.show]: all_meshes[show_to_name[args.show]]}

    config_path = args.config.expanduser().resolve() if args.config is not None else None
    config = _load_config(config_path)
    objects = config["objects"]
    missing = [name for name in all_meshes if name not in objects]
    if missing:
        parser.error(f"Config is missing object entries: {', '.join(missing)}")

    print("\nNOTE: STL does not reliably contain SolidWorks assembly-instance transforms.")
    print("The example/default transforms are all zero; use a measured config for alignment.")
    print("This script loads visual meshes only; it does not simulate collisions or peeling.")
    print("Bounds are computed directly from STL vertices; PyBullet AABB is not used because visual-only bodies have no collision shape.")
    print(f"Display mode: {args.show}; mesh scale: {args.scale}\n")

    client = p.connect(p.GUI, options=f"--width={args.width} --height={args.height}")
    if client < 0:
        raise RuntimeError("Could not connect to PyBullet GUI")

    try:
        p.resetSimulation()
        p.setGravity(0, 0, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        colors = {
            "eye_part_1": [0.86, 0.86, 0.82, 1.0],
            "eye_part_2": [0.25, 0.55, 0.95, 0.65],
            "membrane": [1.0, 0.22, 0.18, 1.0],
        }
        bounds = []
        for name, mesh_path in meshes.items():
            _, world_min, world_max = _add_mesh(name, mesh_path, objects[name], args.scale, colors[name])
            bounds.append((world_min, world_max))

        axis_len = 0.05
        p.addUserDebugLine([0, 0, 0], [axis_len, 0, 0], [1, 0, 0], 2)
        p.addUserDebugLine([0, 0, 0], [0, axis_len, 0], [0, 1, 0], 2)
        p.addUserDebugLine([0, 0, 0], [0, 0, axis_len], [0, 0, 1], 2)
        _fit_camera(bounds)

        print("\nPyBullet preview running. Close the GUI window to exit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / 60.0)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
