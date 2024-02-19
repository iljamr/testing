import matplotlib.pyplot as plt
import numpy as np
from pypcd import PointCloud


def debug_image(img):
    plt.imshow(img)
    plt.show()


def create_range_image(points, sensor_center, range_values, rings):
    # Transform points into local coordinate system
    local_coords = points - sensor_center

    # Create placeholder for range image
    if rings.max() > 32:
        range_image = np.zeros((64, 2048))
    else:
        range_image = np.zeros((32, 2048))

    # Calculates the angular bin
    #
    # OBS! The angles will not match the sensor in terms of rotation (e.g. all
    # sensor positions should be seen as having a fix rotation)
    #
    # OBS! Any correction of lidar points due to rotation during capture could
    # lead to incorrect angle bins
    angles = np.floor(
        (1024 * (np.arctan2(local_coords[:, 1], local_coords[:, 0]) / np.pi + 1)) % 2048
    )
    # Add range values to their correct positions in the range image
    # Note: This could be done for labels and other factors as well
    range_image[rings.astype(int), angles.astype(int)] = range_values

    return range_image


def read_and_project(fn):
    pcd_data = PointCloud.from_path(fn)

    # Points on shape: [Nx3]
    points = np.stack(
        (pcd_data.pc_data["x"], pcd_data.pc_data["y"], pcd_data.pc_data["z"]), axis=1
    )
    obs_vec = np.stack(
        (pcd_data.pc_data["v_x"], pcd_data.pc_data["v_y"], pcd_data.pc_data["v_z"]),
        axis=1,
    )

    if "scan_number" in pcd_data.get_metadata()["fields"]:
        scan_numbers = pcd_data.pc_data["scan_number"]
    else:
        scan_numbers = None

    if "ring" in pcd_data.get_metadata()["fields"]:
        rings = pcd_data.pc_data["ring"]
    else:
        rings = None

    if "range" in pcd_data.get_metadata()["fields"]:
        range_values = pcd_data.pc_data["range"]
        sensor_centers = points + obs_vec * range_values[:, None]
    else:
        range_values = None
        sensor_centers = None

    if "label" in pcd_data.get_metadata()["fields"]:
        labels = pcd_data.pc_data["label"]
    else:
        labels = None

    for scan in np.unique(scan_numbers):
        mask = scan_numbers == scan
        range_image = create_range_image(
            points[mask],
            sensor_centers[mask].mean(axis=0),
            range_values[mask],
            rings[mask],
        )
        range_image
        labels

        # debug_image(range_image)


if __name__ == "__main__":
    pcd_file_path = "/Volumes/mos/data/deepforestry/labled/rev7/2024-01-31-06-28-36_filtered_labeled.pcd"

    read_and_project(pcd_file_path)
