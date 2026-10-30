#!/usr/bin/env python3
"""Diagnose STL meshes for the ILM eye-scene assembly.

Checks:
  * Per-mesh: vertex range, triangle count, surface area, centroid.
  * Pairwise (part1 vs part2): identical geometry up to rigid transform?
  * Membrane: dimensions, shape aspect, and how it sits relative to eye halves.

Pure standard library (struct + re), no third-party deps.
"""
from __future__ import annotations

import argparse
import math
import re
import struct
from pathlib import Path


def read_stl_triangles(path: Path):
    raw = path.read_bytes()
    if len(raw) >= 84:
        tri_count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + tri_count * 50 == len(raw):
            tris = []
            off = 84
            for _ in range(tri_count):
                vals = struct.unpack_from("<12f", raw, off)
                tris.append((tuple(vals[3:6]), tuple(vals[6:9]), tuple(vals[9:12])))
                off += 50
            if tris:
                return tris
    text = raw.decode("utf-8", errors="ignore")
    verts = [
        (float(x), float(y), float(z))
        for x, y, z in re.findall(
            r"\bvertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)",
            text, flags=re.IGNORECASE,
        )
    ]
    if len(verts) % 3 != 0:
        raise ValueError(f"Unsupported ASCII STL: vertex count {len(verts)} not multiple of 3")
    return [(verts[i], verts[i + 1], verts[i + 2]) for i in range(0, len(verts), 3)]


def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _norm(a): return math.sqrt(_dot(a, a))


def _center(tris):
    sx = sy = sz = 0.0; n = 0
    for v0, v1, v2 in tris:
        for v in (v0, v1, v2):
            sx += v[0]; sy += v[1]; sz += v[2]; n += 1
    return (sx / n, sy / n, sz / n)


def _bounds(tris):
    xs = [v[0] for tri in tris for v in tri]
    ys = [v[1] for tri in tris for v in tri]
    zs = [v[2] for tri in tris for v in tri]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _surface_area(tris):
    total = 0.0
    for v0, v1, v2 in tris:
        total += 0.5 * _norm(_cross(_sub(v1, v0), _sub(v2, v0)))
    return total


def _dedup_vertices(tris, tol=1e-6):
    f = round(1.0 / tol)
    out = set()
    for v0, v1, v2 in tris:
        for v in (v0, v1, v2):
            out.add((round(v[0] * f) / f, round(v[1] * f) / f, round(v[2] * f) / f))
    return out


def _orientation_consistency(tris):
    c = _center(tris); agree = total = 0
    for v0, v1, v2 in tris:
        n = _cross(_sub(v1, v0), _sub(v2, v0)); m = _norm(n)
        if m < 1e-12: continue
        n = (n[0]/m, n[1]/m, n[2]/m)
        mid = ((v0[0]+v1[0]+v2[0])/3 - c[0],
               (v0[1]+v1[1]+v2[1])/3 - c[1],
               (v0[2]+v1[2]+v2[2])/3 - c[2])
        if _dot(n, mid) > 0: agree += 1
        total += 1
    return agree / total if total else 0.0


def mesh_report(label, path):
    print(f"\n{'='*60}\nMESH: {label}\nfile : {path}\n{'='*60}")
    tris = read_stl_triangles(path)
    lo, hi = _bounds(tris)
    dims = (hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2])
    area = _surface_area(tris)
    c = _center(tris)
    orient = _orientation_consistency(tris)

    print(f"  triangles            : {len(tris)}")
    print(f"  raw vertices (3/tri) : {len(tris)*3}")
    print(f"  unique vertices      : {len(_dedup_vertices(tris))}")
    print(f"  bounds min (raw u)   : [{lo[0]:.6f}, {lo[1]:.6f}, {lo[2]:.6f}]")
    print(f"  bounds max (raw u)   : [{hi[0]:.6f}, {hi[1]:.6f}, {hi[2]:.6f}]")
    print(f"  bounding dims (mm)   : [{dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f}]"
          f"  (assumes STL unit = mm)")
    print(f"  centroid (raw)       : [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    print(f"  surface area (mm²)   : {area:.3f}")
    print(f"  orientation agree    : {orient:.4f}  (1.0 = consistently outward)")
    return tris, lo, hi, dims, c


def check_rigid_equivalence(tris_a, tris_b, label_a, label_b):
    print(f"\n--- Rigid equivalence: {label_a} vs {label_b} ---")
    na, nb = len(tris_a), len(tris_b)
    print(f"  triangle count  : {na} vs {nb}", end="")
    if na != nb:
        print("  [DIFFERENT -> cannot be rigid transform of each other]"); return
    print("  [SAME]")

    ca = _center(tris_a); cb = _center(tris_b)
    shift = (ca[0]-cb[0], ca[1]-cb[1], ca[2]-cb[2])
    print(f"  centroid {label_a}   : ({ca[0]:.6f}, {ca[1]:.6f}, {ca[2]:.6f})")
    print(f"  centroid {label_b}   : ({cb[0]:.6f}, {cb[1]:.6f}, {cb[2]:.6f})")
    print(f"  translation  B -> A  : ({shift[0]:.6f}, {shift[1]:.6f}, {shift[2]:.6f})")

    va = _dedup_vertices(tris_a)
    vb_shifted = _dedup_vertices(
        (( (v[0]+shift[0], v[1]+shift[1], v[2]+shift[2]) for v in tri) for tri in tris_b))
    overlap = len(va & vb_shifted); union = len(va | vb_shifted)
    print(f"  unique verts A       : {len(va)}")
    print(f"  unique verts B+shift : {len(vb_shifted)}")
    print(f"  vertex overlap       : {overlap} / {union}  "
          f"(Jaccard = {overlap/union:.4f})")
    if overlap / union > 0.95:
        print(f"  >>> {label_b} is a pure TRANSLATE of {label_a} (no rotation)")
    else:
        print(f"  >>> Translate alone is NOT enough; may need rotation + translate, or geometries differ.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part1", required=True, type=Path)
    ap.add_argument("--part2", required=True, type=Path)
    ap.add_argument("--membrane", required=True, type=Path)
    args = ap.parse_args()

    tris1, lo1, hi1, dims1, c1 = mesh_report("eye_part_1", args.part1)
    tris2, lo2, hi2, dims2, c2 = mesh_report("eye_part_2", args.part2)
    trisM, loM, hiM, dimsM, cM = mesh_report("membrane", args.membrane)

    check_rigid_equivalence(tris1, tris2, "eye_part_1", "eye_part_2")

    print(f"\n{'='*60}\nMEMBRANE vs EYE HALVES — relative placement\n{'='*60}")
    eye_lo = (min(lo1[0],lo2[0]), min(lo1[1],lo2[1]), min(lo1[2],lo2[2]))
    eye_hi = (max(hi1[0],hi2[0]), max(hi1[1],hi2[1]), max(hi1[2],hi2[2]))
    eye_dims = (eye_hi[0]-eye_lo[0], eye_hi[1]-eye_lo[1], eye_hi[2]-eye_lo[2])
    eye_center = ((eye_lo[0]+eye_hi[0])/2, (eye_lo[1]+eye_hi[1])/2, (eye_lo[2]+eye_hi[2])/2)

    print(f"  combined eye bounds (mm)  : [{eye_lo[0]:.3f},{eye_lo[1]:.3f},{eye_lo[2]:.3f}] .. [{eye_hi[0]:.3f},{eye_hi[1]:.3f},{eye_hi[2]:.3f}]")
    print(f"  combined eye dims  (mm)   : [{eye_dims[0]:.3f}, {eye_dims[1]:.3f}, {eye_dims[2]:.3f}]")
    print(f"  combined eye center (mm)  : [{eye_center[0]:.3f}, {eye_center[1]:.3f}, {eye_center[2]:.3f}]")
    print(f"\n  membrane bounds (mm)       : [{loM[0]:.3f},{loM[1]:.3f},{loM[2]:.3f}] .. [{hiM[0]:.3f},{hiM[1]:.3f},{hiM[2]:.3f}]")
    print(f"  membrane dims  (mm)       : [{dimsM[0]:.3f}, {dimsM[1]:.3f}, {dimsM[2]:.3f}]")
    print(f"  membrane center (mm)      : [{cM[0]:.3f}, {cM[1]:.3f}, {cM[2]:.3f}]")

    off = (cM[0]-eye_center[0], cM[1]-eye_center[1], cM[2]-eye_center[2])
    print(f"\n  membrane center vs combined eye center (mm) : [{off[0]:.3f}, {off[1]:.3f}, {off[2]:.3f}]")
    fits_x = loM[0] >= eye_lo[0]-1e-3 and hiM[0] <= eye_hi[0]+1e-3
    fits_z = loM[2] >= eye_lo[2]-1e-3 and hiM[2] <= eye_hi[2]+1e-3
    print(f"  membrane fits inside eye in X : {'YES' if fits_x else 'NO'}")
    print(f"  membrane fits inside eye in Z : {'YES' if fits_z else 'NO'}")

    print(f"\n{'='*60}\nASSEMBLY HINTS\n{'='*60}")
    print("""
  * If part1 & part2 are pure TRANSLATES, their assembly offset is simply
    2 x (center_b - center_a) — use that as a starting guess for --part2.position_m.

  * If they need rotation, STL alone does not encode SolidWorks mates.
    Export the transform from SolidWorks or measure angle/distance directly.

  * The membrane sits BETWEEN the eye halves.  Use its Y offset vs combined
    eye center as --membrane.position_m[1].

  * All STLs are in millimetres.  preview_eye_scene.py applies scale=0.001
    when loading into PyBullet (metres).
""")


if __name__ == "__main__":
    main()