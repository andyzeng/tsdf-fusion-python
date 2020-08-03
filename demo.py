"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np

import fusion


ext_imgs = ['.color.jpg', '.color.png']
ext_depth = '.depth.png'
ext_metadata = '.metadata.json'
ext_tsdf = '.tsdf.npy'
ext_col_grid = '.col_grid.npy'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a mesh using TSDF fusion of multiple frames from depth camera')
    parser.add_argument(
        "-d", "--dir_data", required=True,
        help="The directory containing the data to process"
    )
    parser.add_argument(
        "-o", "--dir_output", required=True,
        help="The directory to which to save the output TSDDF volumes and grids"
    )
    parser.add_argument(
        "-b", "--batch_size", required=True, type=int,
        help="The number of images per scene"
    )
    parser.add_argument(
        "-v", "--voxel_size", type=float, default=0.004,
        help="The size of each voxel, in meters"
    )
    parser.add_argument(
        "-i", "--init_tsdf_vol_value", type=float, default=-1.0,
        help="The value with which to initialize TSDF volume. Recommended: +1.0 or -1.0"
    )
    parser.add_argument(
        "-c", "--calc_vol_bounds", action="store_true",
        help="If given, will automatically calulate the size of voxel volume. Else will use fixed volume size"
    )
    parser.add_argument(
        "--disable_gpu", action="store_true",
        help="If given, will only use CPU"
    )
    args = parser.parse_args()
    imgs_per_scene = args.batch_size
    voxel_size = args.voxel_size
    init_tsdf_vol_value = args.init_tsdf_vol_value

    dir_data = Path(args.dir_data)
    if not dir_data.is_dir():
        raise ValueError(f'Not a dir: {dir_data}')
    dir_output = Path(args.dir_output)
    if not dir_output.is_dir():
        print(f'Creating output dir: {dir_output}')
        dir_output.mkdir(parents=True)

    files_color = []
    for ext in ext_imgs:
        files_color += sorted(dir_data.glob('*' + ext))
    files_depth = sorted(dir_data.glob('*' + ext_depth))
    files_metadata = sorted(dir_data.glob('*' + ext_metadata))

    n_files_color = len(files_color)
    n_files_depth = len(files_depth)
    n_files_metadata = len(files_metadata)
    if n_files_color != n_files_depth or n_files_color != n_files_metadata:
        raise ValueError(f'Found unequal number of color ({n_files_color}), depth({n_files_depth})'
                         f'and metadata ({n_files_metadata}) files')

    # Compute TSDF for each scene
    num_scenes = n_files_color // imgs_per_scene
    for idx_scene in range(num_scenes):
        # Compute the 3D bounds in world coordinates of the convex hull of all camera view frustums in the dataset
        if args.calc_vol_bounds:
            print("Estimating voxel volume bounds...")
            vol_bnds = np.zeros((3, 2))
            for idx_img in range(imgs_per_scene):
                img_num = idx_scene * imgs_per_scene + idx_img

                # Read metadata file
                f_metadata = files_metadata[img_num]
                with open(f_metadata) as json_file:
                    metadata = json.load(json_file)
                cam_intr = np.array(metadata["camera"]["intrinsics"])

                # Read depth image and camera pose
                f_depth = files_depth[img_num]
                depth_im = cv2.imread(str(f_depth), cv2.IMREAD_UNCHANGED).astype(float)
                depth_im /= 1000.  # Convert depth to meters. It is saved in 16-bit PNG as millimeters.

                # Compute camera view frustum and extend convex hull
                cam_pose = np.array(metadata["camera"]["transform_cam2world"]["mat_4x4"])
                view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
        else:
            vol_bnds = np.array((
                (-0.256, 0.256),
                (-0.256, 0.256),
                (0.0, 0.256 * 2),
            ))

        print('vol_bnds: ', vol_bnds)

        # Initialize voxel volume
        print("Initializing voxel volume...")
        use_gpu = False if args.disable_gpu else True
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size, init_tsdf_vol_value=init_tsdf_vol_value, use_gpu=use_gpu)

        # Integrate voxels: Loop through RGB-D images and fuse them together
        t0_elapse = time.time()
        for idx_img in range(imgs_per_scene):
            print("Fusing frame %d/%d" % (idx_img + 1, imgs_per_scene))
            img_num = idx_scene * imgs_per_scene + idx_img

            # Read RGB-D image
            f_color = files_color[img_num]
            print('filename_rgb: ', f_color)
            color_image = cv2.imread(str(f_color))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            f_depth = files_depth[img_num]
            depth_im = cv2.imread(str(f_depth), cv2.IMREAD_UNCHANGED).astype(float)
            depth_im /= 1000.  # Convert depth to meters. It is saved in 16-bit PNG as millimeters.

            # Read camera data
            f_metadata = files_metadata[img_num]
            with open(f_metadata) as json_file:
                metadata = json.load(json_file)
            cam_intr = np.array(metadata["camera"]["intrinsics"])
            cam_pose = np.array(metadata["camera"]["transform_cam2world"]["mat_4x4"])

            # Integrate observation into voxel volume (assume color aligned with depth)
            tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

        fps = imgs_per_scene / (time.time() - t0_elapse)
        print("Average FPS: {:.2f}".format(fps))

        # Get the TSDF volume
        tsdf_grid, color_grid = tsdf_vol.get_volume()

        f_tsdf_grid = dir_output / f'scene-{idx_scene:06d}.tsdf{ext_tsdf}'
        np.save(str(f_tsdf_grid), tsdf_grid)
        print(f'Saving TSDF {f_tsdf_grid}')

        f_color_grid = dir_output / f'scene-{idx_scene:06d}.col_grid{ext_col_grid}'
        np.save(str(f_color_grid), color_grid)
        print(f'Saving Color Voxel Grid {f_color_grid}')

        # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
        verts, faces, norms, colors = tsdf_vol.get_mesh()
        f_mesh = dir_output / f'scene-{idx_scene:06d}.mesh.ply'
        fusion.meshwrite(str(f_mesh), verts, faces, norms, colors)
        print(f'Saving mesh to {f_mesh}')

        # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
        # print("Saving point cloud to pc.ply...")
        # point_cloud = tsdf_vol.get_point_cloud()
        # fusion.pcwrite("pc.ply", point_cloud)
