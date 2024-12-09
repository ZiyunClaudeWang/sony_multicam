import argparse
from pathlib import Path
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from metrics import align_point_clouds


def plot_pcd(xyz, color="r"):
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers', marker=dict(size=1.0,color=color))

def align(scan, gt_pcl, debug=False):
    best_error = float("inf")
    best_aligned = None
    best_transform = None

    init_rot_list = generate_rotation_candidates()
    for i, init_rot in enumerate(init_rot_list):
        init_transform = np.eye(4)
        init_transform[:3, :3] = init_rot.as_matrix()
        aligned, reg_obj, gt_pc_tensor = \
            align_point_clouds(  # feature_based_alignment(
                scan,
                gt_pcl,
                trans_init=init_transform,
                voxel_size=1.2,
                verbose=False,
            )

        if reg_obj.inlier_rmse < best_error:
            best_error = reg_obj.inlier_rmse
            best_aligned = aligned#.cpu().numpy()
            best_transform = reg_obj.transformation.cpu().numpy()


    if debug:
        num_points = 10000
        scan_transformed = scan @ best_transform[:3, :3] + best_transform[:3, 3].reshape((-1,3))
        scatter1 = plot_pcd(scan[::len(scan) // num_points], color="green")
        scatter2 = plot_pcd(gt_pcl[::len(gt_pcl) // num_points], color="blue")
        scatter3 = plot_pcd(scan_transformed[::len(scan_transformed) // num_points], color="red")

        fig = go.Figure(data=[scatter1, scatter2, scatter3])
        fig.show()


    return best_aligned, best_transform

def align_pcs(pc1, pc2, debug=True):
    # return transform T_12
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=pcd2, target=pcd1, max_correspondence_distance=0.05,
        init=np.eye(4)
    )

    print(reg_p2p)

    # Extract the rotation matrix from the transformation matrix
    R = reg_p2p.transformation[:3, :3]
    T = reg_p2p.transformation[:3, 3]

    pc1_transformed = (pc1-T.reshape((-1, 3))) @ R

    # Create a 3D scatter plot
    if debug:
        #import pdb;pdb.set_trace()
        num_points = 10000
        scatter1 = plot_pcd(pc1[::len(pc1)//num_points], color="green")
        scatter2 = plot_pcd(pc2[::len(pc2)//num_points], color="blue")
        scatter3 = plot_pcd(pc1_transformed[::len(pc1_transformed)//num_points], color="red")

        fig = go.Figure(data=[scatter1, scatter2, scatter3])
        fig.show()

    return np.eye(4)

def filter_scan(scan, depth_axis, min_depth, max_depth):
    # filter points based on depth
    depth = np.dot(scan, depth_axis)
    mask = np.full(shape=(len(scan),), fill_value=True, dtype=bool)
    if min_depth > 0:
        mask = mask & (depth > min_depth)
    if max_depth > 0:
        mask = mask & (depth < max_depth)
    return scan[mask]


def generate_rotation_candidates():
    rotations = [R.identity()]
    axes = [np.array([0, 1, 0]), np.array([1, 0, 0])]
    angles = [np.pi / 2, np.pi, 3 * np.pi / 2]

    for i in range(len(axes)):
        axis = axes[i]
        for j in range(len(angles)):
            if not (i == 1 and j == 1):
                angle = angles[j]
                rotation_matrix = R.from_rotvec(axis * angle)
                rotations.append(rotation_matrix)

    return rotations





if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Finds the relative transformation between CAD frame and April Tags Frame, 
                                        by aligning the projector PCs expressed in the april tags frame with the CAD model.""")
    parser.add_argument('--scans', type=Path)
    parser.add_argument('--cad-file', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--use-num-pc', type=int, default=10)
    parser.add_argument('--max-depth', type=int, default=-1)
    parser.add_argument('--min-depth', type=int, default=-1)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(str(args.cad_file))  # Read the point cloud
    pc_cad = np.asarray(pcd.points) / 1000

    pcs_april = np.concatenate([np.load(f)['pc'] for f in sorted(list(args.scans.glob("*.npz")))[100:101]])

    pcs_april = filter_scan(
        pcs_april,
        depth_axis=np.array([0, 0, 1], dtype=float),
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )

    _, pose_cad_april_tags = align(pcs_april * 1000, pc_cad * 1000, debug=True)

    np.save(args.output, pose_cad_april_tags)


