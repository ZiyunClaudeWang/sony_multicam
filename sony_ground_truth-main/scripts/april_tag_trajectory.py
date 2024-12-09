import numpy as np
import os
from tqdm import tqdm
import pdb
import yaml
import sys
sys.path.append("/usr/local/lib/python3.11/site-packages/")
from pathlib import Path
import cv2
# from sss.utils.kalibr import load_kalibr
from apriltag import apriltag
from scipy.spatial.transform import Rotation as R
# import tqdm

from viz_pyrender import render


def undistort(image, intrinsics, distortion_parameters, new_intrinsics=None):
    image_undistorted = cv2.undistort(image, intrinsics, distortion_parameters, None, new_intrinsics)
    return image_undistorted

def detect_april_tags(image):
    # tagStandard41h12
    detector = apriltag("tagStandard41h12")
    detections = detector.detect(image)
    return detections


def path_to_us(f: Path):
    # they are in nanoseconds
    return int(f.name.split(".")[0]) / 1e3

def april_tag_id_to_3d_points(id, width=0.01):
    # order of 2D points: lb-rb-rt-lt
    # convention
    #  (0,0,0)  ---> x
    # |  lt--------rt     ---------
    # |  |         |     |         |
    # v  |    0    |     |   1     |
    # y  lb--------rb     ---------
    #
    #     ---------       ---------
    #    |    2    |     |    3    |
    #    |         |     |         |
    #     ---------       ---------

    lt_offsets = np.array([[0, width],
                  [width, width],
                  [width, 0],
                  [0, 0]])

    idx_offset = np.array([[0, 0],
                  [2*width, 0],
                  [0, 2*width],
                  [2*width, 2*width]])

    xy = idx_offset[id] + lt_offsets
    xyz = np.concatenate((xy, np.zeros_like(xy[:,:1])), axis=1)

    return xyz


def draw_on_image(image, results):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r["lb-rb-rt-lt"]
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r["center"][0]), int(r["center"][1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = str(r['id'])#.tag_family.decode("utf-8")

        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        points_3d = april_tag_id_to_3d_points(r['id'])
        for (x,y,z), (x_px, y_px) in zip(points_3d, r['lb-rb-rt-lt']):
            cv2.putText(image, str((x,y)), (int(x_px), int(y_px) - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    return image

def compute_camera_pose(detections, points_3d, camera_matrix):
    ret, rvecs, tvecs = cv2.solvePnP(points_3d, detections, camera_matrix, (0,0,0,0), flags=cv2.SOLVEPNP_ITERATIVE)

    T = np.eye(4)
    T[:3, :3] = rodrigues(rvecs[:,0])
    T[:3,  3] = tvecs[:,0]

    return T, ret

def convert_np(struct):
    if type(struct) == list:
        struct = np.array(struct)
    elif type(struct) == dict:
        struct = {k:convert_np(v) for k,v in struct.items()}
    return struct

def load_kalibr(fn):
    with open(fn, 'r') as f:
        kalibr_dict = yaml.safe_load(f)

    kalibr_dict = convert_np(kalibr_dict)

    for k,v in kalibr_dict.items():
        v['K'] = np.eye(3)
        v['K'][0,0] = v['intrinsics'][0]
        v['K'][1,1] = v['intrinsics'][1]
        v['K'][0,2] = v['intrinsics'][2]
        v['K'][1,2] = v['intrinsics'][3]

    return kalibr_dict

def rodrigues(k):
    # Ensure k is a unit vector
    theta = np.linalg.norm(k)
    if theta < 1e-10:
        return np.eye(3)

    k = k / theta
    kx, ky, kz = k

    # Skew-symmetric matrix K
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])

    # Identity matrix
    I = np.eye(3)

    # Rodrigues' rotation matrix
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

    return R

def store_trajectory_in_tum_format(timestamps_us, poses, output_path):
    timestamps_s = timestamps_us / 1e6
    quaternions = R.from_matrix(poses[:,:3,:3]).as_quat()
    translations = poses[:,:3,3]

    tum_output = np.concatenate((timestamps_s[:,None], translations, quaternions), axis=1)

    np.savetxt(output_path, tum_output)

def parse_results(results):
    pixels = np.concatenate([r["lb-rb-rt-lt"] for r in results if r['id'] < 4])
    points_3d = np.concatenate([april_tag_id_to_3d_points(r['id']) for r in results if r['id'] < 4])
    return pixels, points_3d


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("""""")
    parser.add_argument("--images_folder", type=Path)
    parser.add_argument("--calibration_file", type=Path)
    parser.add_argument("--detections_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--debug_folder", type=Path)
    parser.add_argument("--mesh_path", type=Path)

    parser.add_argument("--recons_path", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    path = args.images_folder
    calibration_file = args.calibration_file

    kalibr = load_kalibr(calibration_file)

    new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(kalibr['cam0']['K'],
                                                        kalibr['cam0']['distortion_coeffs'],
                                                        kalibr['cam0']['resolution'],
                                                        1, kalibr['cam0']['resolution'])

    images_paths = sorted(list(path.glob("*.jpg")))
    timestamps = np.array([path_to_us(f) for f in images_paths])

    poses = []
    timestamps_filtered = []
    detections = []
    points_3d = []

    
    import trimesh
    mesh = trimesh.load_mesh(str(args.mesh_path))
    mesh.vertices = mesh.vertices / 1000

    # mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.01)
    # mesh.vertices = mesh.vertices - np.array([0, 0, 1])

    T_tag_obj = np.array([[-6.16498551e-01, -7.85963343e-01,  4.68098286e-02,
         3.08745591e-02],
       [ 7.86534755e-01, -6.17480888e-01, -8.96833014e-03,
         5.37743120e-02],
       [ 3.59529533e-02,  3.12885946e-02,  9.98863559e-01,
         7.27931424e-05],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])

    # T_tag_obj_debug = np.eye(4)
    # T_tag_obj_debug[:3, :3] = T_tag_obj[:3, :3].T
    # T_tag_obj = np.linalg.inv(T_tag_obj)

    write_to_debug_folder = args.debug_folder is not None
    if write_to_debug_folder:
        args.debug_folder.mkdir(exist_ok=True, parents=True)
        detect_dir = args.debug_folder / "detect"
        detect_dir.mkdir(exist_ok=True, parents=True)

        gt_folder = args.debug_folder / "gt"
        gt_folder.mkdir(exist_ok=True, parents=True)
    
    plot_recons = args.recons_path is not None
    if plot_recons:
        all_recons_ts = []

        ts_files_folder = args.recons_path.parent / str(args.recons_path.name).replace("lut", "depth")

        for file in sorted(list(ts_files_folder.glob("*.png"))):
            ts = int(file.name.split(".")[0])
            all_recons_ts.append(ts)
        all_recons_ts = np.array(all_recons_ts)
        all_recons_ts += args.offset
        frame_ts_in_ns = timestamps * 1e3

        # find the closest
        closest_recons_id = np.searchsorted(all_recons_ts, frame_ts_in_ns)
        pcd_files = sorted(list(args.recons_path.glob("*.npz")))


    images = []

    # for f in tqdm(images_paths):
    #     image = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    #     images.append(image)
    # images = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in images_paths]

    counter = 0
    prev_pose = None
    all_speeds = []
    for j, image_path in enumerate(tqdm(images_paths)):

        # if j < 1000:
        #     continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        images.append(image)

        image = undistort(image, kalibr['cam0']['K'], kalibr['cam0']['distortion_coeffs'], new_intrinsics)
        results = detect_april_tags(image)
        pcd_idx = closest_recons_id[j]

        if len(results) == 0:
            continue

        if plot_recons:
            pc = np.load(pcd_files[pcd_idx])['pc']

        # 4 by N
        pc = np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1).T
        # T_gt_event = np.eye(4)
        T_event_gt = kalibr['cam1']['T_cn_cnm1']
        T_gt_events = np.linalg.inv(T_event_gt)
        pc_in_gt = (T_gt_events @ pc)[:3, :].T

        # project points
        pc_in_events_2d = cv2.projectPoints(pc_in_gt, np.zeros(3), np.zeros(3), new_intrinsics, None)[0].squeeze()

        pc_in_events_2d[:, 0] = np.clip(pc_in_events_2d[:, 0], 0, image.shape[1] - 1)
        pc_in_events_2d[:, 1] = np.clip(pc_in_events_2d[:, 1], 0, image.shape[0] - 1)

        det, landmarks = parse_results(results)
        camera_pose, success = compute_camera_pose(det, landmarks, new_intrinsics)

        if prev_pose is None:
            prev_pose = camera_pose

        speed = np.linalg.norm(prev_pose[:3, 3] - camera_pose[:3, 3]) * 120
        all_speeds.append(speed)
        # print(np.linalg.norm(prev_pose[:3, 3] - camera_pose[:3, 3]) * 120)

        prev_pose = camera_pose

        detections.append(det)
        points_3d.append(landmarks)

        if not success:
            print("No success")
            continue

        # if j % 10 != 0:
        #     continue

        #print(camera_pose)
        if write_to_debug_folder:
            image_drawn = draw_on_image(image, results)
            cv2.imwrite(str(detect_dir / f"{counter:05d}_tag.jpg"), image_drawn)

            new_k = [new_intrinsics[0, 0], new_intrinsics[1, 1], new_intrinsics[0, 2], new_intrinsics[1, 2]]
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            new_mesh = mesh.copy()
            new_mesh.apply_transform(T_tag_obj)
            image_render = render(new_mesh, 
                                    camera_translation=camera_pose[:3,3], 
                                    camera_rotation=camera_pose[:3,:3], 
                                    intrinsic=new_k, 
                                    image=image_color/255.)

            image_render = (image_render * 255).astype(np.uint8)


            cv2.imwrite(str(gt_folder / f"{counter:05d}.jpg"), image_render)

            image_render[pc_in_events_2d[:, 1].astype(int), pc_in_events_2d[:, 0].astype(int)] = [0, 0, 255]
            cv2.imwrite(str(args.debug_folder / f"{counter:05d}.jpg"), image_render)
            # cv2.imshow("image", image_render)

            counter += 1
            # cv2.waitKey(0)

        timestamps_filtered.append(timestamps[j])
        poses.append(camera_pose)

    np.save(os.path.join(args.debug_folder, "speeds.npy"), np.array(all_speeds))
    timestamps_filtered = np.array(timestamps_filtered)
    poses = np.stack(poses, axis=0)

    timestamps_detections = np.concatenate([np.full(shape=(len(d),), fill_value=t) for d, t in zip(detections, timestamps_filtered)])
    detections = np.concatenate(detections)
    points_3d = np.concatenate(points_3d)
    detection_data = np.concatenate((timestamps_detections.reshape((-1,1)), detections, points_3d), axis=-1)
    np.save(str(args.detections_path), detection_data)

    if args.output_path is not None:
        store_trajectory_in_tum_format(timestamps_filtered, poses, args.output_path)
    


