import numpy as np
import open3d as o3d
import trimesh as tri
from scipy.spatial.transform import Rotation as R
from metrics import (
    draw_registration_result,
    align_point_clouds,
    feature_based_alignment,
)


def read_scan_pcl(fpath):
    fh = np.load(fpath)
    return fh["pc"]


def filter_scan(scan, depth_axis, min_depth, max_depth):
    # filter points based on depth
    depth = np.dot(scan, depth_axis)
    mask = np.logical_and(depth > min_depth, depth < max_depth)
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


if __name__ == "__main__":

    test_num = 0

    if test_num == 0:
        # test alignment for a single scan

        # TODO: find proper scaling factor and why it is even needed
        raw_scan = read_scan_pcl("point_clouds/lut_pc_spam_scans/00042.npz")
        scaled_scan = raw_scan * 1000
        scan = filter_scan(
            scaled_scan,
            depth_axis=np.array([0, 0, 1], dtype=float),
            min_depth=140,
            max_depth=170,
        )
        rot = R.from_euler("xyz", [np.pi, 0, 0])
        scan_centroid = np.mean(scan, axis=0)
        scan = (scan - scan_centroid) @ rot.as_matrix()
        scan += scan_centroid

        gt_mesh = tri.load_mesh("point_clouds/spam.ply")
        gt_pcl = np.array(gt_mesh.vertices)

        # move pointclouds so that the GT is centered at zero
        gt_pcl -= np.mean(gt_pcl, axis=0)
        scan -= np.mean(gt_pcl, axis=0)

        init_rot_list = generate_rotation_candidates()

        best_error = float("inf")
        best_aligned = None
        best_rot_idx = -1
        for i, init_rot in enumerate(init_rot_list):
            init_transform = np.eye(4)
            init_transform[:3, :3] = init_rot.as_matrix()
            aligned, reg_obj, gt_pc_tensor = (
                align_point_clouds(  # feature_based_alignment(
                    scan,
                    gt_pcl,
                    trans_init=init_transform,
                    voxel_size=1.2,
                    verbose=False,
                )
            )  # TODO: find proper voxel size
            gt_pts = gt_pc_tensor.point.positions.cpu().numpy()

            # default visualization
            v_scan = scan.copy()
            v_aligned = aligned.copy()
            v_gt_pts = gt_pts.copy()

            # improve visualization angle for disk writes
            view_rot = R.from_euler("xyz", [0, np.pi / 4, np.pi / 4])
            v_scan = view_rot.apply(scan)
            v_aligned = view_rot.apply(aligned)
            v_gt_pts = view_rot.apply(gt_pts)

            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(v_scan)
            aligned_pcd = o3d.geometry.PointCloud()
            aligned_pcd.points = o3d.utility.Vector3dVector(v_aligned)
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(v_gt_pts)

            draw_registration_result(
                scan_pcd,
                aligned_pcd,
                gt_pcd,
                output_file=f"vis/registration_{i}.png",
            )

            print(f"Error for rotation index {i}: {reg_obj.inlier_rmse}")

            if reg_obj.inlier_rmse < best_error:
                best_error = reg_obj.inlier_rmse
                best_aligned = aligned_pcd
                best_rot_idx = i

        print(f"Best rotation index: {best_rot_idx} @ {best_error}")

        o3d.io.write_point_cloud("point_clouds/spam_real_aligned.ply", best_aligned)