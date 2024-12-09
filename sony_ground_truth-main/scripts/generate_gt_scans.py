import argparse
from pathlib import Path
from evo.tools import file_interface
import numpy as np
import open3d as o3d
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--cad-file', type=Path)
    parser.add_argument('--pose-cad-static', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    traj = file_interface.read_tum_trajectory_file(str(args.trajectory))
    pose_a_cad = np.linalg.inv(np.load(args.pose_cad_static))

    pcd = o3d.io.read_point_cloud(str(args.cad_file))
    pc_cad = np.asarray(pcd.points) / 1000 # cad pc is in mm

    # transform cad model into april tags frame
    pc_a = pc_cad @ pose_a_cad[:3,:3].T + pose_a_cad[:3,3].T

    # apply poses of trajectory, and save pc in a separate folder
    args.output.mkdir(exist_ok=True, parents=True)

    for i, (t, pose_ca) in tqdm.tqdm(enumerate(zip(traj.timestamps, traj.poses_se3)), total=len(traj.poses_se3)):
        pc_c = pc_a @ pose_ca[:3,:3].T + pose_ca[:3,3].T
        timestamps = np.full(shape=(len(pc_c,)), fill_value=t)
        np.savez(args.output / f'{i:05d}.npz', timestamps=timestamps, pc_c=pc_c)