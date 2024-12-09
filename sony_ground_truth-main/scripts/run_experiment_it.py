import tqdm


from sss.utils import projector, kalibr, sync, lut, matching, ICP, metrics

import argparse
import yaml
from collections import namedtuple
import os
import open3d as o3d

import glob
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm


def convert_array_to_pil(depth_map, vmin=None, vmax=None):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    mask = depth_map!=0
    disp_map = 1/depth_map

    if vmax is None:
        vmax = np.percentile(disp_map[mask], 95)
    if vmin is None:
        vmin = np.percentile(disp_map[mask], 5)

    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
    mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 0
    return colormapped_im

def draw_registration_result(source, target, transformation):
    import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates point clouds from the events.")

    parser.add_argument("--folder")
    parser.add_argument("--output_folder")
    parser.add_argument("--file", default=None, type=str)
    parser.add_argument("--kalibr-path", default=None, type=str)

    parser.add_argument("--doe-file", dest="doe_path", default='calibration/DOE_extended.npy')
    parser.add_argument("--n-channels", dest="n_channels", type=int, default=32)
    parser.add_argument("--sleep-ms", dest="sleep_ms", type=int, default=1)
    parser.add_argument("--channel-id", dest="channel_id", type=int, default=-1)
    parser.add_argument("--plane-depth", dest="plane_depth", type=float, default=0.19)

    parser.add_argument("--dry-run", dest="dry_run", type=bool, default=False)

    args = parser.parse_args()

    if args.folder is None:
        args.folder = os.path.dirname(args.file)

    with open(os.path.join(args.folder, 'experiment.yaml'), 'r') as f:
        experiment_args = yaml.safe_load(f)

        experiment_args = {k:(os.path.join(args.folder, v) if 'path' in k else v) for k,v in experiment_args.items()}
        ExpArgs = namedtuple("ExpArgs", experiment_args.keys())
        experiment_args = ExpArgs(*experiment_args.values())

    event_data = {}

    event_data["DOE"] = projector.loadExtendedDOE(args.doe_path)

    kalibr_path = args.kalibr_path
    if not os.path.exists(kalibr_path):
        kalibr_path = list(glob.glob(os.path.join(args.folder, "*.yaml")))[0]

    if args.file is not None:
        npy_raw_path = args.file
    else:
        npy_raw_path = experiment_args.npy_raw_path
        if not os.path.exists(npy_raw_path):
            npy_raw_paths = list(glob.glob(os.path.join(args.folder, "*.raw")))
            npy_raw_path = npy_raw_paths[0]

    event_data["kalibr"] = kalibr.load_kalibr(kalibr_path)

    depth = experiment_args.plane_depth
    K = event_data["kalibr"]["cam0"]["K"]
    D = event_data["kalibr"]["cam0"]["distortion_coeffs"]
    res = event_data["kalibr"]["cam0"]["resolution"]
    print(res)

    cam_T_projector = event_data["kalibr"]["cam0"]["T_cam_projector"]

    lut_dict = lut.create_lut(event_data["DOE"], depth, cam_T_projector, K, D, res)
    cam_pix_u = lut_dict["cam_pix_u"]
    # lut.visualize_lut(lut_dict['copy'])

    #total_scans = event_data['frame_ids'].max() / args.n_channels
    #total_time = (event_data['events']['t'].max() - event_data['events']['t'].min()) / 1e6
    #print( "Total scans captured ",  total_scans )
    #print( "Total time captured ",  total_time )
    #print( "Effective frame rate ", total_scans / total_time )

    frame_id = experiment_args.start_frame

    # create folders
    lut_folder = os.path.join(args.output_folder, "lut")
    bf_folder = os.path.join(args.output_folder, "bf")
    sgbm_folder = os.path.join(args.output_folder, "sgbm")

    os.makedirs(lut_folder, exist_ok=True)
    os.makedirs(bf_folder, exist_ok=True)
    os.makedirs(sgbm_folder, exist_ok=True)


    frame_times = sync.get_frame_times_from_file(npy_raw_path, args.n_channels, args.sleep_ms)
    pbar = tqdm.tqdm(total=len(frame_times))

    for channel_ids, events_between_triggers in sync.EventsBetweenSyncIterator(npy_raw_path, frame_times=frame_times, n_frames=args.n_channels):

        events = events_between_triggers[events_between_triggers['y'] < experiment_args.crop_y]
        channel_ids = channel_ids[events_between_triggers['y'] < experiment_args.crop_y]

        if len(events) == 0:
            break

        copy_ids = lut.copy_lookup(
            events, channel_ids, lut_dict["copy"]
        )

        if args.dry_run:
            print(args)
            print(events.shape)
            print(event_data['triggers'].shape)
            print('Dry-run mode enabled - exiting.')
            exit()

        lut_depth = lut.create_depth_image(events, channel_ids, lut_dict['depth'], res)
        lut_pc, _ = lut.create_pc(events, channel_ids, lut_dict['depth'], cam_pix_u, K, return_numpy=True)

        path = os.path.join(lut_folder, "%05d.npz" % frame_id)
        np.savez(path, pc=lut_pc, t=events['t'], Zmin=lut_dict['Zmin'], Zmax=lut_dict['Zmax'])

        if False:
            lut_depth_mask = lut_depth != 0
            lut_min_disp = K[0, 0] * cam_T_projector[0, 3] / np.max(lut_depth[lut_depth_mask])
            lut_max_disp = K[0, 0] * cam_T_projector[0, 3] / np.min(lut_depth[lut_depth_mask])

            lut_depth_mask = lut_depth != 0
            lut_min_disp = K[0, 0] * cam_T_projector[0, 3] / np.max(lut_depth[lut_depth_mask])
            lut_max_disp = K[0, 0] * cam_T_projector[0, 3] / np.min(lut_depth[lut_depth_mask])

            print("LUT Disparity Bounds ", lut_min_disp, lut_max_disp)
            print("Exp Disparity Bounds ", experiment_args.min_disparity, experiment_args.max_disparity)


            print("Creating channel copy volumes")
            cc_vol = matching.create_cc_volume(
                events,
                channel_ids,
                copy_ids,
                res,
                args.n_channels,
            )

            ref_vol = matching.create_proj_volume(
                event_data["DOE"], depth, cam_T_projector, K, D, [1280 + 640, 720], width=experiment_args.laser_width
            )

            min_disparity = 10 * int(np.floor(lut_min_disp / 10)) - 10
            max_disparity = 10 * int(np.ceil(lut_max_disp / 10)) + 10

            print("Disparity Bounds ", min_disparity, max_disparity)

            print("Volume filter")
            filtered = np.zeros_like(cc_vol)
            SSSProj().volume_filter_or(cc_vol, filtered, experiment_args.prefilter_window, False)

            print("Bruteforce")
            bf_disparity, bf_cost = matching.bruteforce(filtered, ref_vol,
                                                                     window = experiment_args.post_filter_window,
                    min_disparity = min_disparity,
                    max_disparity = max_disparity,
                    return_cost = True)
            bf_depth = matching.depth_from_result(cam_pix_u, bf_disparity, cam_T_projector, K, res)
            bruteforce_pc = matching.pc_from_depth(cam_pix_u, bf_depth, cam_T_projector, K, res)

            print("SGBM")
            sgbm_disparity = matching.sgbm(filtered, ref_vol,
                                            window = experiment_args.post_filter_window,
                    min_disparity = min_disparity,
                    max_disparity = max_disparity,
                    return_cost = False)

            sgbm_depth = matching.depth_from_result(cam_pix_u, sgbm_disparity, cam_T_projector, K, res)
            sgbm_pc = matching.pc_from_depth(cam_pix_u, sgbm_depth, cam_T_projector, K, res)

            np.savez(os.path.join(sgbm_folder, "%05d.npz" % frame_id), pc=sgbm_pc, t=events['t'])
            np.savez(os.path.join(bf_folder, "%05d.npz" % frame_id), pc=bruteforce_pc, t=events['t'])


        frame_id += 1
        pbar.update(1)
