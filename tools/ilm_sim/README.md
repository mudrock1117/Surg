# ILM eye-scene preview

This folder adds a **visual-only** PyBullet preview for the two eye-section STL components and the separately exported membrane. It does not modify the existing SurgVLA tasks or robot code.

## Diagnostics added

- `--show all|part1|part2|membrane`: display all meshes or one STL at a time.
- The script prints each loaded mesh's world-space AABB and approximate dimensions in meters and millimeters.
- The camera automatically targets the combined bounding-box center and adjusts its distance to the displayed geometry.
- `--width` and `--height` control the PyBullet GUI dimensions.

## Important coordinate note

STL normally stores triangle geometry, not the SolidWorks assembly-instance transform. Exporting a component by opening its part document can therefore lose the component's placement in the assembly. The script intentionally does not guess those transforms.

- `position_m`: object translation in PyBullet world coordinates, in meters.
- `euler_deg`: XYZ Euler rotation angles, in degrees.
- `--scale 0.001`: default assumes STL coordinates are millimeters and converts them to meters. Confirm the original SolidWorks export units before using this value.
- The checked-in `ilm_scene_config.json` contains the measured transforms for the two instances of the same `Human Eye2.SLDPRT` part and the membrane. The two eye entries intentionally reuse the same local STL geometry; their distinct assembly transforms place the two instances.
- Only visual meshes are loaded; raw STL meshes are not used for collision. Collision and peel mechanics are later steps.

## Run

From the repository root, with the SurgVLA Conda environment activated:

```bash
python tools/ilm_sim/preview_eye_scene.py \
  --part1 "/path/to/Human Eye2.STL" \
  --part2 "/path/to/Human Eye2.1.STL" \
  --membrane "/path/to/membrane.STL" \
  --config tools/ilm_sim/ilm_scene_config.json
```

On Windows PowerShell, quote paths containing spaces:

```powershell
python tools/ilm_sim/preview_eye_scene.py `
  --part1 "D:\models\Human Eye2.STL" `
  --part2 "D:\models\Human Eye2.1.STL" `
  --membrane "D:\models\membrane.STL" `
  --config tools/ilm_sim/scene_config.example.json
```

### Isolate each mesh

Run one command at a time to see which STL produces which geometry:

```powershell
python tools/ilm_sim/preview_eye_scene.py --part1 "D:\models\Human Eye2.STL" --part2 "D:\models\Human Eye2.1.STL" --membrane "D:\models\membrane.STL" --show part1
python tools/ilm_sim/preview_eye_scene.py --part1 "D:\models\Human Eye2.STL" --part2 "D:\models\Human Eye2.1.STL" --membrane "D:\models\membrane.STL" --show part2
python tools/ilm_sim/preview_eye_scene.py --part1 "D:\models\Human Eye2.STL" --part2 "D:\models\Human Eye2.1.STL" --membrane "D:\models\membrane.STL" --show membrane
```

Replace the example paths with the actual local paths. The script still requires all three paths to exist even when displaying one mesh.

Copy `scene_config.example.json` to your own config file and replace the placeholder transforms with the assembly transforms from SolidWorks. Keep the original STL files unchanged.


## Transform validation

The preview treats `rotation_matrix` as authoritative and verifies that it is a proper
rotation and agrees with `euler_deg` under SciPy's XYZ convention. It fails early
if the two representations disagree. The current eye-part-2 matrix is valid; do not
change its signs. This preview is a visual check, not proof of watertight seam contact
or membrane adhesion.

A transform-aware, coarse AABB check can be run with:

```bash
python tools/ilm_sim/validate_scene_config.py \
  --part1 "/path/to/Human Eye2.STL" \
  --part2 "/path/to/Human Eye2.1.STL" \
  --membrane "/path/to/membrane.STL" \
  --config tools/ilm_sim/ilm_scene_config.json
```

The AABB report is only a broad-phase spatial check. It cannot establish exact surface
contact or the anatomical location of the membrane.
