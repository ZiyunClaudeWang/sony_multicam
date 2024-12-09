# To run on the cluster, use:
#   $ source /mnt/kostas-graid/sw/envs/ioannis/torchlight/bin/activate

import time
import torch
import numpy as np
import open3d as o3d
import trimesh as tri
import os
from scipy.spatial.transform import Rotation as R
import copy
from pathlib import Path

DATA_ROOT = Path("point_clouds/")


def load_ply_file(ply_file, asarray=True):
    """Load a .ply file from disk.

    Args:
        ply_file (str): file path to the .ply file
        asarray (bool, optional): whether to return the data as a NumPy array. Defaults to True.

    Returns:
        Open3D point cloud or NumPy array of points: the content of the .ply file
    """
    pcd = o3d.io.read_point_cloud(str(ply_file))
    if asarray:
        return np.asarray(pcd.points)
    return pcd


def file_to_vis(
    ply_file, points_sampled=10000, output_file="output.png", save_sampled_ply=False
):
    """Visualize a point cloud using Open3D. Does not save to disk unless OpenGL is available.

    Args:
        ply_file (str): Path to data file on disk
        points_sampled (int, optional): How many points to render on screen. Defaults to 10000.
        output_file (str, optional): Path to save the visualization to. Defaults to "output.png".
    """
    pcd = load_ply_file(ply_file, asarray=False)
    fname = os.path.splitext(os.path.basename(ply_file))[0]
    inds = np.random.choice(len(pcd.points), points_sampled, replace=False)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[inds])
    if save_sampled_ply:
        save_path = DATA_ROOT / f"{fname}_sampled_{points_sampled}.ply"
        o3d.io.write_point_cloud(str(save_path), pcd)
    o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_plotly([pcd])
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # # vis.capture_screen_image(output_file)  # doesn't work
    # vis.destroy_window()


def align_point_clouds(
    partial_scan,
    gt_scan,
    max_dist=500,
    trans_init=np.eye(4),
    voxel_size=None,
    verbose=False,
):
    """Aligns two point clouds using ICP.

    Args:
        partial_scan (numpy array): N x 3 array of points from sensor
        gt_scan (numpy array): M x 3 array of points from ground truth
        max_dist (float, optional): Max distance for ICP to search. Defaults to 500.
        trans_init (numpy array, optional): Initial guess for ICP. Defaults to np.eye(4).
        voxel_size (float, optional): Voxel size for downsampling. Defaults to None.
        verbose (bool, optional): Print debug information. Defaults to False.

    Returns:
        (aligned_partial_scan_np, reg_p2p): aligned partial scan and the corresponding registration object
    """
    ground_truth_pcd = o3d.t.geometry.PointCloud(np.array(gt_scan)).cuda()
    partial_pcd = o3d.t.geometry.PointCloud(partial_scan).cuda()

    ground_truth_pcd.transform(trans_init)

    if voxel_size is not None:
        partial_pcd = partial_pcd.voxel_down_sample(voxel_size=voxel_size)
        ground_truth_pcd = ground_truth_pcd.voxel_down_sample(voxel_size=voxel_size)

    if verbose:
        draw_t_pcd_pair(partial_pcd, ground_truth_pcd)
    dawn = time.time()

    reg_p2p = o3d.t.pipelines.registration.icp(
        partial_pcd,
        ground_truth_pcd,
        max_dist,
        np.eye(4, dtype=float),
        # TODO: assuming normals are available/computable, could use point-to-plane
        o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=150),
    )

    dusk = time.time()
    if verbose:
        print(
            f"ICP took {dusk - dawn:.3f} seconds for {len(partial_pcd.point.positions)} points to align to {len(ground_truth_pcd.point.positions)} points."
        )

    aligned_partial_scan = partial_pcd.transform(reg_p2p.transformation)
    aligned_partial_scan_np = aligned_partial_scan.point.positions.cpu().numpy()

    return aligned_partial_scan_np, reg_p2p, ground_truth_pcd


def draw_registration_result(original, fixed, target, output_file=None):
    """Draw registration results using Open3D.

    Args:
        original (open3d pointcloud): direct sensor data
        fixed (open3d pointcloud): aligned sensor data
        target (open3d pointcloud): ground truth data (subsample before passing for performance)
    """
    source_temp = copy.deepcopy(original)
    fixed_temp = copy.deepcopy(fixed)
    target_temp = copy.deepcopy(target)

    source_temp.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.0, 0.0], (len(source_temp.points), 1))
    )
    fixed_temp.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 0.0], (len(fixed_temp.points), 1))
    )
    target_temp.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.0, 1.0], (len(target_temp.points), 1))
    )

    if output_file is None:
        o3d.visualization.draw_geometries([source_temp, fixed_temp, target_temp])
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(source_temp)
        vis.add_geometry(fixed_temp)
        vis.add_geometry(target_temp)
        # ctr = vis.get_view_control()
        # parameters = ctr.convert_to_pinhole_camera_parameters()
        # parameters.extrinsic = np.array(
        #     [
        #         [0.7071, -0.7071, 0, 0],
        #         [0.7071, 0.7071, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1],
        #     ]
        # )
        # ctr.convert_from_pinhole_camera_parameters(parameters)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_file)
        vis.destroy_window()

    ## TODO: this is somehow broken?
    # origin_frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.5, origin=[0, 0, 0]
    # )
    # o3d.visualization.draw_geometries(
    #     [source_temp, fixed_temp, target_temp, origin_frame]
    # )


def draw_t_pcd_pair(pcd_1, pcd_2):
    """Draw two TENSOR point clouds using Open3D.

    Args:
        pcd_1 (open3d pointcloud): first TENSOR point cloud
        pcd_2 (open3d pointcloud): second TENSOR point cloud
    """
    p1 = o3d.geometry.PointCloud()
    p2 = o3d.geometry.PointCloud()

    p1.points = o3d.utility.Vector3dVector(pcd_1.point.positions.cpu().numpy())
    p2.points = o3d.utility.Vector3dVector(pcd_2.point.positions.cpu().numpy())
    p1.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.0, 1.0], (len(p1.points), 1))
    )
    p2.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.0, 0.0], (len(p2.points), 1))
    )

    o3d.visualization.draw_geometries([p1, p2])


def feature_based_alignment(source, target, voxel_size=3.2):
    """Align two point clouds using feature-based alignment.

    Args:
        source (open3d pointcloud): source point cloud
        target (open3d pointcloud): target point cloud

    Returns:
        (aligned_source, reg_p2p): aligned source point cloud and the corresponding registration object
    """
    source = o3d.t.geometry.PointCloud(source)  # .cuda()
    target = o3d.t.geometry.PointCloud(target)  # .cuda()
    if voxel_size is not None:
        source = source.voxel_down_sample(voxel_size=voxel_size)
        target = target.voxel_down_sample(voxel_size=voxel_size)

    source.estimate_normals(max_nn=30, radius=0.1)
    target.estimate_normals(max_nn=30, radius=0.1)

    source_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(source)  # dim=33
    target_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(target)

    dawn = time.time()

    ## RANSAC-based approach - crashes/hangs with no logs
    reg_p2p = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        max_correspondence_distance=0.05,
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=2,
        # checkers=[
        #     o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
        # ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=100
        ),
    )

    ## Fast global registration - unavailable?
    # reg_p2p = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
    #     source,
    #     target,
    #     source_fpfh,
    #     target_fpfh,
    #     o3d.registration.FastGlobalRegistrationOption(
    #         maximum_correspondence_distance=0.05
    #     ),
    # )

    dusk = time.time()
    print(f"Feature-based alignment took {dusk - dawn:.3f} seconds.")

    aligned_source = source.transform(reg_p2p.transformation)
    return aligned_source, reg_p2p


def compute_metric(proj_fname, gt_fname, gt_subs_fname=None):
    """Computes a distance from a projected point cloud to a ground truth point cloud.

    Args:
        proj_fname (str): path to sensor data file
        gt_fname (str): path to ground truth data file
        gt_subs_fname (str, optional): path to subsampled ground truth file, for visualization only. Defaults to None.

    Returns:
        dist: mean of shortest distances between the sensor point cloud and the ground truth mesh
    """
    proj_ply = DATA_ROOT / proj_fname
    gt_ply = DATA_ROOT / gt_fname
    proj_pcd = load_ply_file(proj_ply)
    gt_pcd = load_ply_file(gt_ply)
    aligned, _ = align_point_clouds(proj_pcd, gt_pcd)

    if gt_subs_fname is not None:
        orig_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(proj_pcd)
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(aligned)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_subs_fname)
        draw_registration_result(orig_pcd, aligned_pcd, gt_pcd)

    gt_mesh = tri.load_mesh(DATA_ROOT / gt_fname)
    (closest_points, distances, triangle_id) = gt_mesh.nearest.on_surface(aligned)

    return np.mean(distances)


if __name__ == "__main__":
    test_num = 2

    if test_num == 0:
        # Test loading of a simple ply file
        fname = "tuna.ply"
        ply_file = DATA_ROOT / fname
        ply_array = load_ply_file(ply_file)
        print(ply_array.shape)
        print(ply_array[:10])

        file_to_vis(ply_file, points_sampled=1000, save_sampled_ply=True)

    elif test_num == 1:
        # Test mesh distance computation
        fname = "mesh.ply"
        mesh = tri.load_mesh(DATA_ROOT / fname)
        verts = mesh.vertices
        inds = verts[:, 2] > np.mean(verts[:, 2])
        partial_pcd = verts[inds] + np.random.normal(0, 0.01, verts[inds].shape)

        (closest_points, distances, triangle_id) = mesh.nearest.on_surface(partial_pcd)

        print(distances.shape)
        print(distances[:10])
        print(np.histogram(distances))
        dist_eval = np.mean(distances)
        print(f"EVALUATION METRIC: {dist_eval:.3f}")
        print("-" * 20)
        print(partial_pcd[:20])
        print(closest_points[:20])
        print("-" * 20)

    elif test_num == 2:
        # Test ICP alignment
        fname = "tuna.ply"
        mesh = tri.load_mesh(DATA_ROOT / fname)
        verts = mesh.vertices
        chop_axis = 0
        inds = verts[:, chop_axis] > np.mean(verts[:, chop_axis])
        partial_pcd = verts[inds] + np.random.normal(0, 0.5, verts[inds].shape)

        rot = R.from_euler("xyz", np.random.rand(3) * np.pi / 10, degrees=False)
        print(f"Rotation matrix: {rot.as_matrix()}")
        rot_partial_pcd = rot.apply(partial_pcd)
        tr_partial_pcd = rot_partial_pcd + np.array([[4, -2, 1]])

        aligned, reg_obj = align_point_clouds(tr_partial_pcd, partial_pcd)

        orig_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(tr_partial_pcd)
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(aligned)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(verts)
        draw_registration_result(orig_pcd, aligned_pcd, gt_pcd)

        o3d.io.write_point_cloud(
            str(DATA_ROOT / f"spam_chaos_original_{chop_axis}.ply"),
            orig_pcd,
        )
        o3d.io.write_point_cloud(
            str(DATA_ROOT / f"spam_chaos_aligned_{chop_axis}.ply"),
            aligned_pcd,
        )

    elif test_num == 3:
        # Test end-to-end pipeline
        gt_fname = "mesh.ply"
        proj_fname = f"tr_partial_random_10000_0.ply"

        dist = compute_metric(proj_fname, gt_fname)
        print(f"Done, mean distance: {dist:.3f}")