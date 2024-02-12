import numpy as np

# from numpy.lib.recfunctions import structured_to_unstructured
import tqdm
from pathlib import Path
from . import pypcd as pypcd
import numba


def get_metadata(fields, size, types, count):
    # Construct dtype for each point
    data_fields = {}
    data_types = []
    bytes_per_point = 0
    for f, s, t, c in zip(
        fields.split(" ")[1:],
        size.split(" ")[1:],
        types.split(" ")[1:],
        count.split(" ")[1:],
    ):
        if t == "F":
            if s == "8":
                data_type = np.float64
            elif s == "4":
                data_type = np.float32
            elif s == "2":
                data_type = np.float16
        elif t == "U":
            if s == "8":
                data_type = np.uint64
            elif s == "4":
                data_type = np.uint32
            elif s == "2":
                data_type = np.uint16
            elif s == "1":
                data_type = np.uint8
        elif t == "I":
            if s == "8":
                data_type = np.int64
            elif s == "4":
                data_type = np.int32
            elif s == "2":
                data_type = np.int16
            elif s == "1":
                data_type = np.int8

        data_types.append((f, data_type))
        bytes_per_point += int(s)
        data_fields[f] = {
            "nbytes": int(s) * int(c),
            "size": int(s),
            "count": int(c),
            "type": data_type,
        }
    return data_fields, data_types, bytes_per_point


def merge_pcd_file(pcd_file, dir_out, length=5.0, suffix="*_segmented.pcd"):
    in_dir = Path(dir_out)
    tmp_out_file = open(pcd_file.with_suffix(".pcd.tmp"), "wb")

    n_points = 0
    for i, file in tqdm.tqdm(enumerate(list(in_dir.glob(suffix))), desc="tiles"):
        in_file = open(file, "rb")
        # Assumes that all files have the same data types and fields
        # Read header information
        tmp = in_file.readline().decode().strip()
        if "#" in tmp:
            comment = tmp
            version = in_file.readline().decode().strip()
        else:
            # No comment in current file
            version = tmp
            comment = "# Merged by streamed_pcd_tiling "
        # version = in_file.readline().decode().strip()
        fields = in_file.readline().decode().strip()
        size = in_file.readline().decode().strip()
        types = in_file.readline().decode().strip()
        count = in_file.readline().decode().strip()
        _ = in_file.readline().decode().strip()  # width
        _ = in_file.readline().decode().strip()  # height
        viewpoint = in_file.readline().decode().strip()
        points = in_file.readline().decode().strip()
        data = in_file.readline().decode().strip()

        data_fields, data_types, bytes_per_point = get_metadata(
            fields, size, types, count
        )

        current_tile_idx = file.stem.split("_")[1:3]
        for _ in tqdm.trange(int(points.split(" ")[-1]), mininterval=1, desc="points"):
            pt_raw = in_file.read(bytes_per_point)
            pt = np.frombuffer(pt_raw, dtype=data_types)
            xy = np.array([pt["x"], pt["y"]]).T

            ti = (xy // length).astype(int).flatten()
            tile_idx = "{}_{}".format(ti[0], ti[1])
            if tile_idx == "{}_{}".format(current_tile_idx[0], current_tile_idx[1]):
                n_points += 1
                tmp_out_file.write(pt_raw)
    tmp_out_file.close()

    if n_points == 0:
        raise RuntimeError("No points to be merged, final segmentation file is invalid")

    with open(pcd_file.with_suffix(".pcd.tmp"), "rb") as pts_file:
        with open(pcd_file, "wb") as tile_file:
            tile_file.write("{}\n".format(comment).encode())
            tile_file.write("{}\n".format(version).encode())
            tile_file.write("{}\n".format(fields).encode())
            tile_file.write("{}\n".format(size).encode())
            tile_file.write("{}\n".format(types).encode())
            tile_file.write("{}\n".format(count).encode())

            # Update weight and height (Ignore possible structure)
            tile_file.write("WIDTH {}\n".format(n_points).encode())
            tile_file.write("HEIGHT 1\n".encode())

            tile_file.write("{}\n".format(viewpoint).encode())

            # Update number of points
            tile_file.write("POINTS {}\n".format(n_points).encode())

            tile_file.write("{}\n".format(data).encode())

            for i in range(n_points):
                tile_file.write(pts_file.read(bytes_per_point))

    # Remove the tmp files
    pcd_file.with_suffix(".pcd.tmp").unlink()


@numba.njit()
def filter_points_simple(
    xy, point_scan_idx, point_range, pose, scan_idx, offsets, length, simple=True
):
    if np.isnan(xy).any():
        return None

    tile_indices = (xy[:, :2] + offsets) // length  # .astype(int)
    return xy, tile_indices


def tile_pcd_file(
    pcd_file, dir_out, length=5.0, buffer=1.0, trajectory_file=None, invert_z_axis=False
):
    assert (
        buffer < length
    ), "Warning: buffer cannot be smaller than length in current implementation"
    assert length > 0, "Warning: length should be above zero"
    assert buffer >= 0, "Warning, buffer should be zero or positive"
    in_file = open(pcd_file, "rb")

    out_dir = Path(dir_out)
    out_dir.mkdir(exist_ok=True)

    # Read header information
    comment = in_file.readline().decode().strip()
    version = in_file.readline().decode().strip()
    fields = in_file.readline().decode().strip()
    size = in_file.readline().decode().strip()
    types = in_file.readline().decode().strip()
    count = in_file.readline().decode().strip()
    _ = in_file.readline().decode().strip()  # width
    _ = in_file.readline().decode().strip()  # height
    viewpoint = in_file.readline().decode().strip()
    points = in_file.readline().decode().strip()
    data = in_file.readline().decode().strip()

    data_fields, data_types, bytes_per_point = get_metadata(fields, size, types, count)

    scan_idx = None
    pose = None
    filter_func = filter_points_simple
    # contiguous = True

    # Read and cluster each point into tiles
    if buffer > 0:
        # Create offsets for buffer
        offsets = (
            np.array(
                [
                    [-1, -1],
                    [-1, 0],
                    [-1, 1],
                    [0, -1],
                    [0, 0],
                    [0, 1],
                    [1, -1],
                    [1, 0],
                    [1, 1],
                ]
            )
            * buffer
        )
    else:
        offsets = np.array([[0, 0]])

    tiles = {}
    for _ in tqdm.trange(int(points.split(" ")[-1]), mininterval=1):
        # Read one point and decode it
        pt_raw = in_file.read(bytes_per_point)
        pt = np.frombuffer(pt_raw, dtype=data_types).copy()

        if invert_z_axis:
            # Invert z axis if z is pointing downwards
            pt["x"] *= -1
            pt["z"] *= -1
            pt_raw = pt.tobytes()

        xy = filter_func(
            np.array([pt["x"], pt["y"], pt["z"]]).T,
            None,  # pt['scan_number'],
            None,  # pt['range'],
            pose,
            scan_idx,
            offsets,
            length,
        )
        if xy is None:
            continue
        else:
            xy, tile_indices = xy

        processed_tiles = []
        for ti in tile_indices.astype(int):
            tile_idx = "{}_{}".format(ti[0], ti[1])
            if tile_idx in processed_tiles:
                continue
            else:
                processed_tiles.append(tile_idx)
            if tile_idx not in tiles:
                tiles[tile_idx] = {
                    "filename": "{}/tile_{}_.pcd".format(dir_out, tile_idx),
                    "file": open("{}/tile_{}_.pcd.tmp".format(dir_out, tile_idx), "wb"),
                    "n_points": 0,
                }

            tiles[tile_idx]["file"].write(pt_raw)
            tiles[tile_idx]["n_points"] += 1

    # Write each tile as a proper .pcd file (header depends on the content)
    for tile in tiles:
        tiles[tile]["file"].close()
        with open(tiles[tile]["filename"] + ".tmp", "rb") as pts_file:
            with open(tiles[tile]["filename"], "wb") as tile_file:
                tile_file.write("{}\n".format(comment).encode())
                tile_file.write("{}\n".format(version).encode())
                tile_file.write("{}\n".format(fields).encode())
                tile_file.write("{}\n".format(size).encode())
                tile_file.write("{}\n".format(types).encode())
                tile_file.write("{}\n".format(count).encode())

                # Update weight and height (Ignore possible structure)
                tile_file.write("WIDTH {}\n".format(tiles[tile]["n_points"]).encode())
                tile_file.write("HEIGHT 1\n".encode())

                tile_file.write("{}\n".format(viewpoint).encode())

                # Update number of points
                tile_file.write("POINTS {}\n".format(tiles[tile]["n_points"]).encode())

                tile_file.write("{}\n".format(data).encode())

                for i in range(tiles[tile]["n_points"]):
                    tile_file.write(pts_file.read(bytes_per_point))

        # Remove the tmp files
        Path(tiles[tile]["filename"] + ".tmp").unlink()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--length", type=float, default=10, help="Max side length of each tile"
    )
    parser.add_argument(
        "--buffer", type=float, default=0.0, help="Overlap between neighbouring tiles"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/karl/Data/DF/2023-08-22-09-37-39/RawMap_cropped.pcd",
        help="File for tiling",
    )
    parser.add_argument(
        "--traj_file", type=str, default=None, help="File sensor positions"
    )
    parser.add_argument(
        "--out_dir", type=str, default="tmp", help="Output directory to save tiles into"
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        help="Reverses the process by mergin the tiles in out_dir",
    )

    args = parser.parse_args()

    if args.merge:
        merge_pcd_file(
            Path(args.input), args.out_dir, length=args.length, suffix="*_.pcd"
        )
    else:
        tile_pcd_file(
            args.input,
            args.out_dir,
            length=args.length,
            buffer=args.buffer,
            trajectory_file=args.traj_file,
        )
