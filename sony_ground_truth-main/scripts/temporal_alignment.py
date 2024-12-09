from argparse import ArgumentParser

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def find_best_correlation(s1, t1, s2, t2):
    # Create a common time grid based on the union of both timestamps
    common_time_grid = np.linspace(
        max(t1.min(), t2.min()),
        min(t1.max(), t2.max()),
        num=1000  # You can adjust the number of points for resolution
    )

    # Interpolate both arrays to the common time grid
    interp1 = interp1d(t1, s1, kind='linear', fill_value="extrapolate")
    interp2 = interp1d(t2, s2, kind='linear', fill_value="extrapolate")

    # Get the interpolated data
    array1_interpolated = interp1(common_time_grid)
    array2_interpolated = interp2(common_time_grid)

    # Compute cross-correlation
    correlation = np.correlate(array1_interpolated - np.mean(array1_interpolated),
                               array2_interpolated - np.mean(array2_interpolated),
                               mode='full')

    # Find the index of the maximum correlation
    max_correlation_index = np.argmax(correlation)

    # Compute the lag in terms of the common grid index
    lag_index = max_correlation_index - (len(array1_interpolated) - 1)
    lag_time = lag_index * (common_time_grid[1] - common_time_grid[0])  # Time lag

    return lag_time

def resample_trajectory(timestamps, trajectory):
    t, T, q = trajectory[:,0], trajectory[:,1:4], trajectory[:,4:]
    mask = (timestamps >= t[0]) & (timestamps <= t[-1])
    timestamps = timestamps[mask]

    interp = interp1d(t, T, kind='linear', fill_value="extrapolate", axis=0)
    T_interp = interp(timestamps)
    q_interp = Slerp(t, R.from_quat(q))(timestamps).as_quat()
    trajectory_resampled = np.concatenate((t[mask,None], T_interp, q_interp), axis=1)
    return trajectory_resampled

def velocity_norm_from_trajectory(trajectory, roll=0):
    t, T = trajectory[:,0], trajectory[:,1:4]
    if roll > 0:
        T = np.roll(T, -roll, axis=0)

    v = np.diff(T, axis=0) / np.diff(t)[:,None]
    v_norm = np.linalg.norm(v, axis=1)
    return v_norm, t[:-1]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Coarsely align trajectory to reference temporally.")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    trajectory = np.genfromtxt(args.trajectory)
    reference = np.genfromtxt(args.reference)

    v_traj_norm, t_traj = velocity_norm_from_trajectory(trajectory)
    v_ref_norm, t_ref = velocity_norm_from_trajectory(reference)

    v_traj_norm = np.convolve(v_traj_norm, np.array([1/3,1/3,1/3]), mode='same')

    lag_time = find_best_correlation(v_traj_norm, t_traj, v_ref_norm, t_ref)

    trajectory_aligned = resample_trajectory(trajectory[:,0] + lag_time, trajectory)

    v_traj_norm_res, t_traj_res = velocity_norm_from_trajectory(trajectory_aligned)

    print(f"Found lag time of {lag_time:.2f} seconds")

    fig, ax = plt.subplots()
    ax.plot(t_traj, v_traj_norm, 'b', label="Robot Gripper Trajectory")
    ax.plot(t_traj_res, v_traj_norm_res, 'r', label="Robot Gripper Temporally Aligned with Events")
    ax.plot(t_ref, v_ref_norm, 'g', label="April Tag Trajectory from Events")
    ax.set_ylabel("Velocity norm")
    ax.set_xlabel("Time [s]")
    ax.legend()
    plt.show()

    np.savetxt(args.output, trajectory_aligned, delimiter=" ")
    print(f"Wrote output to {str(args.output)}")

