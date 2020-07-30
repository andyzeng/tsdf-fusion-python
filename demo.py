"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import argparse
import time

import cv2
import numpy as np

import fusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a mesh using TSDF fusion of multiple frames from depth camera')
    parser.add_argument(
        "-d", "--dir_data", required=True,
        help="The directory containing the data to process"
    )
    parser.add_argument(
        "-b", "--batch_size", required=True,
        help="The number of images per scene"
    )
    parser.add_argument(
        "-t", "--total_frames", type=int, default=1,
        help="The total number of frames to process. Each frame has 1 rgb, 1 depth and 1 pose.txt file"
    )
    parser.add_argument(
        "-c", "--calc_vol_bounds", action="store_true",
        help="If given, will automatically calulate the size of voxel volume. Else will use fixed volume size"
    )
    parser.add_argument(
        "-v", "--voxel_size", type=float, default=0.004,
        help="The size of each voxel, in meters"
    )
    parser.add_argument(
        "--init_tsdf_vol_value", type=float, default=1.0,
        help="The value with which to initialize TSDF volume. Recommended: +1.0 or -1.0"
    )
    args = parser.parse_args()
    n_imgs = args.total_frames
    voxel_size = args.voxel_size
    init_tsdf_vol_value = args.init_tsdf_vol_value

    # Convert Blender's Camera axes to TSDF axes
    rot_x_180 = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    rot_x_neg_90 = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, -1, 0, 0],
                             [0, 0, 0, 1]])
    rot_x_90 = np.array([[1, 0, 0, 0],
                         [0, 0, -1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
    rot_y_90 = np.array([[0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [-1, 0, 0, 0],
                         [0, 0, 0, 1]])
    rot_z_90 = np.array([[0, -1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')

    # noinspection PyShadowingNames
    def get_cam_pose(path_txt):
        cam_pose = np.loadtxt(path_txt)  # 4x4 rigid transformation matrix
        cam_pose = cam_pose @ rot_x_180
        return cam_pose

    # Compute the 3D bounds in world coordinates of the convex hull of all camera view frustums in the dataset
    if args.calc_vol_bounds:
        print("Estimating voxel volume bounds...")
        vol_bnds = np.zeros((3, 2))
        for i in range(n_imgs):
            # Read depth image and camera pose
            filename = "data/frame-%06d.depth.png" % i
            print('filename_depth: ', filename)
            depth_im = cv2.imread(filename, -1).astype(float)
            depth_im[depth_im == 65535] = 0  # Clean depth. Specific to 7-scenes dataset.
            depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters

            cam_pose = get_cam_pose("data/frame-%06d.pose.txt" % i)

            # Compute camera view frustum and extend convex hull
            view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    else:
        vol_bnds = np.array((
            (-0.256, 0.256),
            (-0.256, 0.256),
            (-0.256, 0.256),
        ))

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size, init_tsdf_vol_value=init_tsdf_vol_value, use_gpu=True)

    # Integrate voxels: Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))

        # Read RGB-D image and camera pose
        filename = "data/frame-%06d.color.jpg" % i
        print('filename_rgb: ', filename)
        color_image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("data/frame-%06d.depth.png" % i, -1).astype(float)
        depth_im[depth_im == 65535] = 0  # Clean depth. Specific to 7-scenes dataset.
        depth_im /= 1000.

        cam_pose = get_cam_pose("data/frame-%06d.pose.txt" % i)

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    # print("Saving point cloud to pc.ply...")
    # point_cloud = tsdf_vol.get_point_cloud()
    # fusion.pcwrite("pc.ply", point_cloud)
