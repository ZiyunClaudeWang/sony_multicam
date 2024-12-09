from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse


def quaternion_to_rotation_matrix(quaternion, scalar_first=True):
    # if scalar_first: (w, x, y, z), else: (x, y, z, w)
    r = R.from_quat(quaternion, scalar_first=scalar_first)
    return r.as_matrix()


def rotation_matrix_to_quaternion(matrix, scalar_first=True):
    r = R.from_matrix(matrix)
    return r.as_quat()#scalar_first=scalar_first)


class TransformWrapper:
    def __init__(self, transform, timestamp, frame_id, child_frame_id):
        self.transform = transform
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id

    def __str__(self):
        tf_str = str(self.transform)
        ret = f"{self.timestamp}: {self.frame_id} -> {self.child_frame_id}\n{tf_str}"
        return ret

    def get_tum_str(self):
        # w must be last
        q = rotation_matrix_to_quaternion(self.transform[:3, :3])#, scalar_first=False)
        ret = f"{self.timestamp} {self.transform[0, 3]} {self.transform[1, 3]} {self.transform[2, 3]} {q[0]} {q[1]} {q[2]} {q[3]}"
        return ret

    def compose(self, other):
        # sanity check
        assert self.child_frame_id == other.frame_id
        assert self.timestamp == other.timestamp

        # return composition
        composed_tf = self.transform @ other.transform
        return TransformWrapper(
            composed_tf, other.timestamp, self.frame_id, other.child_frame_id
        )

    def inv(self):
        inv_tf = np.linalg.inv(self.transform)
        return TransformWrapper(
            inv_tf, self.timestamp, self.child_frame_id, self.frame_id
        )


def extract_kinova_tf(msg):
    ret = []
    # loop over joints
    for i in range(len(msg.transforms)):

        # get matrix repr
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = quaternion_to_rotation_matrix(
            np.array(
                [
                    msg.transforms[i].transform.rotation.w,
                    msg.transforms[i].transform.rotation.x,
                    msg.transforms[i].transform.rotation.y,
                    msg.transforms[i].transform.rotation.z,
                ]
            ),
            scalar_first=True,
        )
        tf_mat[:3, 3] = np.array(
            [
                msg.transforms[i].transform.translation.x,
                msg.transforms[i].transform.translation.y,
                msg.transforms[i].transform.translation.z,
            ]
        )

        frame_id = msg.transforms[i].header.frame_id
        child_frame_id = msg.transforms[i].child_frame_id
        tf_timestamp_sec = (
            msg.transforms[i].header.stamp.sec
            + msg.transforms[i].header.stamp.nanosec * 1e-9
        )
        ret.append(TransformWrapper(tf_mat, tf_timestamp_sec, frame_id, child_frame_id))

    return ret


def extract_franka_tf(msg):
    # format is flattened, plus we need to avoid weird read-only buffer error?!
    tf_mat = np.array(msg.O_T_EE.reshape(4, 4).T)
    frame_id = "base"
    child_frame_id = "end_effector"
    tf_timestamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    ret = TransformWrapper(tf_mat, tf_timestamp_sec, frame_id, child_frame_id)
    return ret


def read_single_bagfile(bagpath, arm_used):
    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_FOXY)  # TODO: is this correct?

    topic_name = (
        "/tf"
        if arm_used == "kinova"
        else "/panda/franka_state_controller_custom/franka_states"
    )

    ret = []
    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            if arm_used == "kinova":
                tfw_list = extract_kinova_tf(msg)
                ret.append(tfw_list)
            else:
                tfw_list = extract_franka_tf(msg)
                ret.append(tfw_list)

    return ret


def compile_kinova_tfs(all_tfs):
    # get transform from base to brace across all time steps
    ret = []
    for timestamp_tfs in all_tfs:
        T_base_brace = timestamp_tfs[1]
        for i in range(2, 8):
            T_base_brace = T_base_brace.compose(timestamp_tfs[i])
        ret.append(T_base_brace)
    return ret


def export_as_tum(tf_list, tum_fpath, root_times=False, step=1):
    print(f"Exporting to TUM format: {tum_fpath}")
    base_time = tf_list[0].timestamp
    with open(tum_fpath, "w") as f:
        for i in range(0, len(tf_list), step):
            tf = tf_list[i]
            if root_times:
                tf.timestamp -= base_time
            f.write(tf.get_tum_str() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ROS bag files and export transformations in TUM text format."
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["kinova", "franka"],
        required=True,
        help="Type of robotic arm used.",
    )
    parser.add_argument(
        "--bagfile", type=str, required=True, help="Path to the ROS bag file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output TUM file."
    )
    parser.add_argument(
        "--root-times",
        action="store_true",
        help="Root the timestamps to the first timestamp (first timestamp will be 0).",
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Step size for exporting transformations."
    )
    args = parser.parse_args()

    if args.arm == "kinova":
        all_tfs = read_single_bagfile(Path(args.bagfile), arm_used="kinova")
        T_base_brace_list = compile_kinova_tfs(all_tfs)
        export_as_tum(
            T_base_brace_list, args.output, root_times=args.root_times, step=args.step
        )
    elif args.arm == "franka":
        T_O_EE = read_single_bagfile(Path(args.bagfile), arm_used="franka")
        export_as_tum(T_O_EE, args.output, root_times=args.root_times, step=args.step)
