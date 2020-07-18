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
        "-t", "--total_frames", type=int, default=1,
        help="The total number of frames to process. Each frame has 1 rgb, 1 depth and 1 pose.txt file")
    parser.add_argument(
        "-c", "--calc_vol_bounds", action="store_true",
        help="If given, will automatically calulate the size of voxel volume. Else will use fixed volume size")
    args = parser.parse_args()
    n_imgs = args.total_frames
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')

    # Compute the 3D bounds in world coordinates of the convex hull of all camera view frustums in the dataset
    if args.calc_vol_bounds:
        print("Estimating voxel volume bounds...")
        vol_bnds = np.zeros((3, 2))
        for i in range(n_imgs):
            # Read depth image and camera pose
            filename = "data/frame-%06d.depth.png" % i
            print('filename_depth: ', filename)
            depth_im = cv2.imread(filename, -1).astype(float)
            depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
            depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
            cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % i)  # 4x4 rigid transformation matrix

            # Compute camera view frustum and extend convex hull
            view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    else:
        vol_bnds = np.array((
            (-0.256, 0.256),
            (-0.256, 0.256),
            (0, 0.256),
        ))

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.004 * 1)

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

        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % i)

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
