"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import argparse
import time
import json
from pathlib import Path
import OpenEXR
import Imath
import os

import cv2
import numpy as np

import fusion


ext_imgs = ['.color.jpg', '.color.png']
ext_depth = '.depth.exr'
ext_metadata = '.metadata.json'
ext_tsdf = '.tsdf.npy'
ext_mask = '.mask.npy'
ext_col_grid = '.col_grid.npy'

def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array
    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)
    """
    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()["dataWindow"]
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ["R", "G", "B"]:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)
        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels)

        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel("R", pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr


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

    ext_scenes = "scene-"
    scenes = sorted(dir_data.glob(ext_scenes + '*'))
    num_scenes = len(scenes)
    for idx_scene in range(num_scenes):
        dir_scene = scenes[idx_scene]
        files_color = []
        for ext in ext_imgs:
            files_color += sorted(dir_scene.glob('*' + ext))
        files_depth = sorted(dir_scene.glob('*' + ext_depth))
        files_metadata = sorted(dir_scene.glob('*' + ext_metadata))
        if args.calc_vol_bounds:
            print("Estimating voxel volume bounds...")
            vol_bnds = np.zeros((3, 2))
            for idx_img in range(imgs_per_scene):
                img_num = idx_img

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
                cam_pose = np.array(metadata["camera"]["transforms"]["numpy_cam"]["cam2world"]["mat_4x4"])
                view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
        else:
            vol_bnds = np.array((
                (-0.256, 0.256),
                (-0.256, 0.256),
                (0, 0.256),
            ))

        print('vol_bnds: ', vol_bnds)

        # Initialize voxel volume
        print("Initializing voxel volume...")
        use_gpu = False if args.disable_gpu else True
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size, init_tsdf_vol_value=init_tsdf_vol_value, use_gpu=use_gpu)

        # Integrate voxels: Loop through RGB-D images and fuse them together
        t0_elapse = time.time()

        vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(
            order='C').astype(int)
        mask = np.zeros(vol_dim, dtype=bool)
        for idx_img in range(imgs_per_scene):
            print("Fusing frame %d/%d" % (idx_img + 1, imgs_per_scene))
            img_num = idx_img
            # Read RGB-D image
            f_color = files_color[img_num]
            print('filename_rgb: ', f_color)
            color_image = cv2.imread(str(f_color))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            f_depth = files_depth[img_num]
            depth_im = exr_loader(str(f_depth), 1)
            # Read camera data
            f_metadata = files_metadata[img_num]
            with open(f_metadata) as json_file:
                metadata = json.load(json_file)
            cam_intr = np.array(metadata["camera"]["intrinsics"])
            cam_pose = np.array(metadata["camera"]["transforms"]["numpy_cam"]["cam2world"]["mat_4x4"])

            # Integrate observation into voxel volume (assume color aligned with depth)
            mask = tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, mask, obs_weight=1.)

        fps = imgs_per_scene / (time.time() - t0_elapse)
        print("Average FPS: {:.2f}".format(fps))

        # Get the TSDF volume
        tsdf_grid, color_grid = tsdf_vol.get_volume()
        os.mkdir(dir_output / f'scene-{idx_scene:06d}')
        f_mask = dir_output / f'scene-{idx_scene:06d}' / f'mask{ext_mask}'
        np.save(str(f_mask), mask)
        print(f'Saving Mask {f_mask}')

        f_tsdf_grid = dir_output / f'scene-{idx_scene:06d}' / f'tsdf{ext_tsdf}'
        np.save(str(f_tsdf_grid), tsdf_grid)
        print(f'Saving TSDF {f_tsdf_grid}')

        f_color_grid = dir_output / f'scene-{idx_scene:06d}' / f'col_grid{ext_col_grid}'
        np.save(str(f_color_grid), color_grid)
        print(f'Saving Color Voxel Grid {f_color_grid}')

        # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
        verts, faces, norms, colors = tsdf_vol.get_mesh()
        f_mesh = dir_output / f'scene-{idx_scene:06d}' / f'mesh.ply'
        fusion.meshwrite(str(f_mesh), verts, faces, norms, colors)
        print(f'Saving mesh to {f_mesh}')

