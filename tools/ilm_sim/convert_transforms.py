
import csv
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_components(csv_path):
    components = {}

    with open(csv_path, "r", encoding="gbk", errors="replace", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row["component_name"]
            suppressed = row["suppressed"].strip().lower() == "true"

            if suppressed:
                continue

            values = np.array(
                [float(row[f"m{i}"]) for i in range(16)],
                dtype=float
            )

            R = values[:9].reshape(3, 3)
            t = values[9:12]
            scale = values[12]

            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name}: 矩阵中存在无效数值")

            orth_error = np.linalg.norm(R.T @ R - np.eye(3))
            det = np.linalg.det(R)

            print(f"\n组件: {name}")
            print("平移 (m):", t)
            print("缩放:", scale)
            print("旋转矩阵正交误差:", orth_error)
            print("旋转矩阵行列式:", det)

            if abs(scale - 1.0) > 1e-6:
                print("警告：缩放不是 1，请进一步检查")

            if orth_error > 1e-4 or abs(det - 1.0) > 1e-4:
                raise ValueError(
                    f"{name}: 旋转矩阵异常，请检查矩阵排列或数据"
                )

            euler = Rotation.from_matrix(R).as_euler(
                "xyz", degrees=True
            )

            components[name] = {
                "position_m": t.tolist(),
                "euler_deg": euler.tolist(),
                "rotation_matrix": R.tolist(),
                "scale": float(scale),
            }

    return components


def find_component(components, token):
    matches = [
        name for name in components
        if token.lower() in name.lower()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"标识 '{token}' 匹配到 {len(matches)} 个组件：{matches}。"
            "请根据 CSV 中的 component_name 修改匹配字符串。"
        )

    return components[matches[0]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--part1", required=True)
    parser.add_argument("--part2", required=True)
    parser.add_argument("--membrane", required=True)
    parser.add_argument("--output", default="ilm_scene_config.json")

    args = parser.parse_args()

    components = load_components(args.csv)

    config = {
        "objects": {
            "eye_part_1": find_component(components, args.part1),
            "eye_part_2": find_component(components, args.part2),
            "membrane": find_component(components, args.membrane),
        }
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n配置已生成:", output.resolve())


if __name__ == "__main__":
    main()