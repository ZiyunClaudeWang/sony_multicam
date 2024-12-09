import argparse
from pathlib import Path
from evo.tools import file_interface
import numpy as np
import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from evo.core.trajectory import PoseTrajectory3D
from typing import Union

class Interpolator:
    def __init__(self, trajectory: Union[PoseTrajectory3D, np.array], timestamps=None):
        if type(trajectory) == PoseTrajectory3D:
            timestamps = trajectory.timestamps
            positions = trajectory.positions_xyz
            quaternions = R.from_quat(trajectory.orientations_quat_wxyz, scalar_first=True)
        else:
            assert timestamps is not None
            positions = trajectory[:,:3,3]
            quaternions = R.from_matrix(trajectory[:,:3,:3])

        self.interp = interp1d(timestamps, positions, kind='linear', fill_value="extrapolate", axis=0)
        self.q_interp = Slerp(timestamps, quaternions)

    def __call__(self, timestamps, output_type=PoseTrajectory3D):
        positions = self.interp(timestamps)
        quaternions = self.q_interp(timestamps)

        if output_type == PoseTrajectory3D:
            return PoseTrajectory3D(timestamps=timestamps, orientations_quat_wxyz=quaternions.as_quat(scalar_first=True),
                                    positions_xyz=positions)
        else:
            T = np.zeros((len(timestamps), 4, 4))
            T[:,-1,-1] = 1
            T[:,:3,:3] = quaternions.as_matrix()
            T[:,:3, 3] = positions
            return T

def load_pc(file, crop_by_range=True):
    fh = np.load(file)
    pc = fh['pc']
    t = fh['t']

    if crop_by_range:
        Zmin, Zmax = fh['Zmin'], fh['Zmax']
        Zmin = np.max(Zmin)
        Zmax = np.min(Zmax)
        mask = (pc[:,-1] >= Zmin) & (pc[:,-1] <= Zmax)
        pc = pc[mask]
        t = t[mask]

    return {"pc": pc, "t": t/1e6}


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Remaps scans from the projector in the april tags reference frame.""")
    parser.add_argument("--scans", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    traj = file_interface.read_tum_trajectory_file(str(args.trajectory))
    poses_ac = np.stack([np.linalg.inv(p) for p in traj.poses_se3])
    interpolator = Interpolator(poses_ac, timestamps=traj.timestamps)

    scan_files = sorted(list(args.scans.glob("*.npz")))
    args.output.mkdir(exist_ok=True, parents=True)

    for file in tqdm.tqdm(scan_files):
        data = load_pc(file, crop_by_range=True)
        poses_t = interpolator(timestamps=data['t'], output_type=np.array)
        pc_remapped = np.einsum("nij,nj->ni", poses_t[:,:3,:3], data['pc']) + poses_t[:,:3, 3]
        np.savez(args.output / file.name, pc=pc_remapped, t=data['t'])


