from argparse import ArgumentParser

import scipy.linalg
from evo.core import sync
from evo.core import lie_algebra as lie
from evo.tools import file_interface
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from sss.utils.kalibr import load_kalibr
import cv2
from evo.core.trajectory import PoseTrajectory3D

def twist_to_pose(x):
    w, v = x[:3], x[3:]
    W = np.zeros((4,4))
    W[:3,:3] = np.cross(np.eye(3), w)
    W[:3, 3] = v
    return scipy.linalg.expm(W)

def reprojection_residual(x, reference_trajectory, trajectory, points_3d, detections_normalized):
    # x: 12,
    # reference_trajectory: N x 4 x 4 -> april tags
    # trajectory: N x 4 x 4 -> robot gripper
    # points_3d: N x 3
    # detections_normalized: N x 3

    # camera (c) -> april_tags (a) -> gripper (g) -> robot_base (b) -> camera (c)
    pose_cb = twist_to_pose(x[:6])
    pose_ga = twist_to_pose(x[6:])

    poses_bg = trajectory

    points_3d_hom = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
    points_3d_c = np.einsum("ij,njk,km,nm->ni", pose_cb,poses_bg, pose_ga, points_3d_hom)
    residual = detections_normalized[:,:2] - points_3d_c[:,:2] / points_3d_c[:,-2:-1]

    residual = residual.ravel()

    return residual


def pose_residual(x, reference_trajectory, trajectory):
    # x: 12,
    # reference_trajectory: N x 4 x 4 -> april tags
    # trajectory: N x 4 x 4 -> robot gripper

    # camera (c) -> april_tags (a) -> gripper (g) -> robot_base (b) -> camera (c)
    pose_cb = twist_to_pose(x[:6])
    pose_ga = twist_to_pose(x[6:])

    poses_bg = trajectory
    poses_ac = reference_trajectory

    pose_cc = np.einsum("ij,njk,km,nml->nil", pose_cb, poses_bg, pose_ga, poses_ac)

    twist = np.stack([scipy.linalg.logm(p) for p in pose_cc]) # N x 4 x 4
    residual = np.concatenate((twist[:,:3,3], twist[:,[0,0,1],[1,2,2]]), axis=1)
    residual = residual.ravel()

    return residual

def optimize(trajectory_ref, trajectory, detections_normalized, points_3d, broadcasting):
    # trajectory_ref -> T_ca

    x0 = np.zeros((12,))

    kwargs = dict(
        reference_trajectory=np.stack([np.linalg.inv(T) for T in trajectory_ref.poses_se3]),
        trajectory=np.stack(trajectory.poses_se3)
    )

    result = least_squares(fun=pose_residual, x0=x0, kwargs=kwargs, verbose=2)
    x0 = result.x

    kwargs["detections_normalized"] = detections_normalized
    kwargs["points_3d"] = points_3d
    kwargs["reference_trajectory"] = kwargs["reference_trajectory"][broadcasting]
    kwargs["trajectory"] = kwargs["trajectory"][broadcasting]

    result = least_squares(fun=reprojection_residual, x0=x0, kwargs=kwargs, verbose=2)
    x = result.x

    pose_cb = twist_to_pose(x[:6])
    pose_ga = twist_to_pose(x[6:])

    return pose_cb, pose_ga

def normalize_coordinates(detections, calibration, use_new_camera_matrix=True):
    K = calibration['cam0']['K']
    if use_new_camera_matrix:
        K, _ = cv2.getOptimalNewCameraMatrix(calibration['cam0']['K'],
                                             calibration['cam0']['distortion_coeffs'],
                                             calibration['cam0']['resolution'],
                                             1, calibration['cam0']['resolution'])
    detections_hom = np.concatenate([detections, np.ones((len(detections), 1))], axis=1)
    detections_norm = detections_hom @ np.linalg.inv(K).T

    return detections_norm


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser("Coarsely aligns trajectory to reference")
    parser.add_argument("--trajectory", type=Path, help="Robot odometry trajectory.")
    parser.add_argument("--reference", type=Path, help="April Tags trajectory.")
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--kalibr-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    traj_ref = file_interface.read_tum_trajectory_file(str(args.reference))
    traj = file_interface.read_tum_trajectory_file(str(args.trajectory))
    detections = np.load(args.detections)

    calibration = load_kalibr(args.kalibr_file)


    print("registering and aligning trajectories")
    traj_ref_cut, traj_cut = sync.associate_trajectories(traj_ref, traj)

    detections = detections[np.isin(detections[:, 0] / 1e6, traj_ref_cut.timestamps)]
    t_unique, inverse = np.unique(detections[:, 0], return_inverse=True)
    points_3d = detections[:, -3:]
    detections_px = detections[:,1:3]
    detections_normalized = normalize_coordinates(detections_px, calibration, use_new_camera_matrix=True)

    #(timestamps_detections.reshape((-1, 1)), detections, points_3d), axis = -1)
    pose_cb, pose_ga = optimize(traj_ref_cut, traj_cut, detections_normalized=detections_normalized, points_3d=points_3d, broadcasting=inverse)

    # looking for T_ca we get it from
    poses_bg = np.stack(traj.poses_se3)
    poses_ca = np.einsum("ij,njk,km->nim", pose_cb, poses_bg, pose_ga)
    traj_aligned = PoseTrajectory3D(timestamps=traj.timestamps, poses_se3=list(poses_ca))

    file_interface.write_tum_trajectory_file(str(args.output), traj_aligned)


