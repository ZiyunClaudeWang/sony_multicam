
import pdb
import numpy as np
import matplotlib.pyplot as plt

import sys

import pyrender
import trimesh

sys.path.append("../")

def create_raymond_lights():
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

def render(
            mesh: trimesh.Trimesh,
        # vertices: np.array,
                camera_translation: np.array,
                camera_rotation:np.array,
                intrinsic:np.array,
                image:np.array,
                trans = False,
                rot_angle=None,
                mesh_base_color=(180./255., 180./255., 180./255.),
                scene_bg_color=(0,0,0),
                return_rgba=False,
                opacity=1.0,
                # face = None
                ) -> np.array:
    renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                            viewport_height=image.shape[0],
                                            point_size=1.0)
    # renderer = pyrender.Renderer(viewport_width=image.shape[1],
    #                                         viewport_height=image.shape[0],
    #                                         point_size=1.0)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(*mesh_base_color, opacity))

    # camera_translation[0] *= -1.


    # # mesh = trimesh.Trimesh(vertices.copy(), face.copy())
    # if rot_angle!=None:
    #     rot = trimesh.transformations.rotation_matrix(
    #         np.radians(rot_angle), [0, 1, 0])
    #     mesh.apply_transform(rot)

    x_rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    # mesh.apply_transform(rot)

    # R_cam_obj = trimesh.transformations.rotation_matrix(camera_rotation)
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = camera_rotation[:3, :3]
    T_cam_obj[:3, 3] = camera_translation

    # R_cam_obj = trimesh.transformations.rotation_matrix(camera_rotation) 
    # mesh.apply_transform(R_cam_obj)
    # mesh.apply_translation(camera_translation)
    # mesh.apply_transform(x_rot)
    
    # mesh = trimesh.Trimesh(mesh)
    mesh = mesh.copy()
    mesh.apply_transform(T_cam_obj)
    mesh.apply_transform(x_rot)

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(bg_color=[0., 1.0, 1.0, 0.0],
                            ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    # camera_pose = np.eye(4)
    # camera_pose[:3, 3] = camera_translation
    # camera_pose[:3,:3] = camera_rotation

    # print(intrinsic)
    camera = pyrender.IntrinsicsCamera(fx=intrinsic[0], fy=intrinsic[1],
                                        cx=intrinsic[2], cy=intrinsic[3], zfar=1e12)
    # scene.add(camera, pose=camera_pose)
    scene.add(camera, pose=np.eye(4))

    light_nodes = create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)
    
    # scene.show()
    # pyrender.Viewer(scene)

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    renderer.delete()

    # if return_rgba:
    #     return color

    valid_mask = (color[:, :, -1])[:, :, np.newaxis]
    if not trans:
        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
    else:
        output_img = color[:, :, :]

    # output_img = color
    output_img = output_img.astype(np.float32)
    return output_img