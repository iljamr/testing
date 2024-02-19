import os
import pathlib
import time

import numpy as np
import open3d as o3d
import yaml

RATE_HZ = 10
SEQUENCE = 8
BASE_PATH = pathlib.Path(os.environ["DATA_PATH"])
SEQUENCE_PATH = "kitti/SemanticKITTI/symlinks/motionbev/sequences"
BIN_PATH = BASE_PATH / SEQUENCE_PATH / f"{SEQUENCE:02}/velodyne"
LABEL_PATH = BASE_PATH / SEQUENCE_PATH / f"{SEQUENCE:02}/labels"
CONFIG_PATH = (
    "/Users/ilja/Div/github/testing/scratchpad/notebooks/config/semantic-kitti-mos.yaml"
)


def load_config(config_path):
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
        return config


def load_color_map(config, return_rgb=False):
    color_map = config["color_map"]
    learning_map_inv = config["learning_map_inv"]
    learning_map = config["learning_map"]

    bgr_color_map = {
        key: color_map[learning_map_inv[learning_map[key]]]
        for key, _ in color_map.items()
    }

    if return_rgb:
        return {key: tuple(reversed(color)) for key, color in bgr_color_map.items()}

    return bgr_color_map


def load_labels(path):
    labels = np.fromfile(path, dtype=np.int32)
    semantic_labels = labels & 0xFFFF
    instance_labels = labels >> 16
    return semantic_labels, instance_labels


def load_scan(path):
    return np.fromfile(path, dtype=np.float32).reshape((-1, 4))


def update_point_cloud(pcd, scan, colors):
    pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)


def create_colors(semantic_labels, color_map):
    return (
        np.array([color_map[label] for label in semantic_labels], dtype=np.float64)
        / 255
    )


def update_visualizer(pcd, visualizer):
    visualizer.update_geometry(pcd)
    visualizer.poll_events()
    visualizer.update_renderer()


def combine_labels(semantic_labels, instance_labels):
    labels = (instance_labels << 16) | semantic_labels
    return labels


def main(visualizer):
    first_scan_path, *scan_paths = sorted(BIN_PATH.glob("*.bin"))
    first_label_path, *label_paths = sorted(LABEL_PATH.glob("*.label"))

    config = load_config(CONFIG_PATH)
    color_map = load_color_map(config, return_rgb=True)
    first_scan = load_scan(first_scan_path)
    semantic_labels, _ = load_labels(first_label_path)

    colors = create_colors(semantic_labels, color_map)

    pcd = o3d.geometry.PointCloud()

    update_point_cloud(pcd, first_scan, colors)

    visualizer.create_window()
    visualizer.add_geometry(pcd)

    for scan_path, lable_path in zip(scan_paths, label_paths):
        scan = load_scan(scan_path)
        semantic_labels, _ = load_labels(lable_path)
        colors = create_colors(semantic_labels, color_map)

        update_point_cloud(pcd, scan, colors)

        if pcd.has_colors():
            update_visualizer(pcd, visualizer)

        time.sleep(1 / RATE_HZ)

    visualizer.destroy_window()


if __name__ == "__main__":
    visualizer = o3d.visualization.Visualizer()

    try:
        main(visualizer)
    except KeyboardInterrupt:
        visualizer.destroy_window()
        print("Visualizer closed")
