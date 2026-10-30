#!/usr/bin/env python3
"""Validate ILM scene transforms and report transformed mesh AABBs.

This checks coordinate transforms and coarse spatial placement only. It does not
prove surface contact, watertightness, or anatomical correctness.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + count * 50 == len(raw):
            values = np.empty((count * 3, 3), dtype=np.float64)
            offset = 84
            for i in range(count):
                tri = struct.unpack_from("<12f", raw, offset)
                values[3 * i:3 * i + 3] = np.asarray(tri[3:12]).reshape(3, 3)
                offset += 50
            if len(values):
                return values
    text = raw.decode("utf-8", errors="ignore")
    found = re.findall(
        r"\bvertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)",
        text, re.IGNORECASE,
    )
    if not found:
        raise ValueError(f"No STL vertices found: {path}")
    return np.asarray(found, dtype=np.float64)


def transform_for(name: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    t = np.asarray(cfg.get("position_m", [0, 0, 0]), dtype=np.float64)
    if t.shape != (3,) or not np.all(np.isfinite(t)):
        raise ValueError(f"{name}.position_m must contain 3 finite numbers")
    euler = cfg.get("euler_deg")
    reuler = Rotation.from_euler("xyz", euler, degrees=True).as_matrix() if euler is not None else None
    if "rotation_matrix" in cfg:
        rotation = np.asarray(cfg["rotation_matrix"], dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(f"{name}.rotation_matrix must be a finite 3x3 matrix")
        orth_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
        determinant = float(np.linalg.det(rotation))
        if orth_error > 1e-5 or abs(determinant - 1.0) > 1e-5:
            raise ValueError(
                f"{name}: invalid rotation matrix "
                f"(orthogonality error={orth_error:.3e}, det={determinant:.9f})"
            )
        if reuler is not None and np.max(np.abs(rotation - reuler)) > 1e-5:
            raise ValueError(f"{name}: Euler angles and rotation matrix disagree")
    elif reuler is not None:
        rotation = reuler
    else:
        rotation = np.eye(3)
    object_scale = float(cfg.get("scale", 1.0))
    if not math.isfinite(object_scale) or object_scale <= 0:
        raise ValueError(f"{name}.scale must be positive and finite")
    return rotation, t, object_scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", required=True, type=Path)
    parser.add_argument("--part2", required=True, type=Path)
    parser.add_argument("--membrane", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stl-to-meter", type=float, default=0.001,
        help="Convert raw STL units to meters (default: mm -> m)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.stl_to_meter) or args.stl_to_meter <= 0:
        parser.error("--stl-to-meter must be positive and finite")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    objects = config.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("Config must contain an 'objects' dictionary")

    inputs = {
        "eye_part_1": args.part1,
        "eye_part_2": args.part2,
        "membrane": args.membrane,
    }
    boxes = {}
    for name, path in inputs.items():
        if name not in objects:
            raise KeyError(f"Missing config object: {name}")
        vertices = read_vertices(path)
        rotation, translation, object_scale = transform_for(name, objects[name])
        # Convert STL units to meters, apply per-object scale, rotate, then translate.
        world = (rotation @ (vertices * args.stl_to_meter * object_scale).T).T + translation
        lower, upper = world.min(axis=0), world.max(axis=0)
        boxes[name] = (lower, upper)
        print(f"\n[{name}] {path}")
        print(f"  triangles: {len(vertices) // 3:,}; raw vertices: {len(vertices):,}")
        print(f"  world AABB min (mm): {np.round(lower * 1000, 3).tolist()}")
        print(f"  world AABB max (mm): {np.round(upper * 1000, 3).tolist()}")
        print(f"  AABB size (mm): {np.round((upper - lower) * 1000, 3).tolist()}")
        print(f"  AABB center (mm): {np.round(((lower + upper) / 2) * 1000, 3).tolist()}")

    a0, a1 = boxes["eye_part_1"]
    b0, b1 = boxes["eye_part_2"]
    overlap = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    gap = np.maximum(0.0, np.maximum(a0, b0) - np.minimum(a1, b1))
    print("\n[eye instance AABB relation]")
    print(f"  overlap extent (mm): {np.round(overlap * 1000, 3).tolist()}")
    print(f"  separated-axis gap (mm): {np.round(gap * 1000, 3).tolist()}")
    print("  NOTE: AABB overlap is a coarse broad-phase test, not proof of mesh contact.")
    print("  NOTE: Compare with the SolidWorks assembly before changing any pose.")


if __name__ == "__main__":
    main()
