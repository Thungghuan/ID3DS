# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
import json
import shutil
from typing import List, Optional, Tuple, Union
import sys

sys.path.append("./eg3d")
for root, dirs, files in os.walk("./eg3d"):
    for dir in dirs:
        sys.path.append(os.path.join(root, dir))


import click
from eg3d import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm, trange
from torchvision.utils import save_image

from eg3d import legacy
from eg3d.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from eg3d.torch_utils import misc
from eg3d.training.triplane import TriPlaneGenerator


# ----------------------------------------------------------------------------


def parse_range(s: Union[str, List]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple):
        return s
    parts = s.split(",")
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f"cannot parse 2-vector {s}")


# ----------------------------------------------------------------------------


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


# ----------------------------------------------------------------------------


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    return samples.unsqueeze(0), voxel_origin, voxel_size


# ----------------------------------------------------------------------------


# @click.command()
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
# @click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
# @click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
# @click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# @click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
# @click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
# @click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
# @click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
# @click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str = "pretrained_models/ffhqrebalanced512-128.pkl",
    seeds: List[int] = [1234],
    truncation_psi: float = 1,
    truncation_cutoff: int = 14,
    outdir: str = "dataset",
    shapes: bool = False,
    shape_res: int = 512,
    fov_deg: float = 18.837,
    shape_format: str = ".mrc",
    class_idx: Optional[int] = None,
    reload_modules: bool = False,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = (
            TriPlaneGenerator(*G.init_args, **G.init_kwargs)
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print(
            "Generating multiview images for seed %d (%d/%d) ..."
            % (seed, seed_idx, len(seeds))
        )
        os.makedirs(f"{outdir}/seed{seed:04d}", exist_ok=True)

        result_dir = f"{outdir}/seed{seed:04d}/multiview"
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir, exist_ok=True)

        SAMPLE_Z_NUM = 100000
        z = (
            torch.from_numpy(
                np.random.RandomState(seed).randn(SAMPLE_Z_NUM, 1, G.z_dim)
            )
            .mean(dim=0)
            .to(device)
        )
        torch.save({"z_arg": z}, f"{outdir}/seed{seed:04d}/z_avg.pkl")

        frame_idx = 0

        imgs = []
        for angle_p, angle_y in tqdm(
            torch.cartesian_prod(
                torch.linspace(0.4, -0.4, 8),
                torch.linspace(-0.8, 0.8, 12),
            )
        ):
            frame_idx += 1
            cam_pivot = torch.tensor(
                G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device
            )
            cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
            cam2world_pose = LookAtPoseSampler.sample(
                np.pi / 2 + angle_y,
                np.pi / 2 + angle_p,
                cam_pivot,
                radius=cam_radius,
                device=device,
            )
            conditioning_cam2world_pose = LookAtPoseSampler.sample(
                np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device
            )
            intrinsics = FOV_to_intrinsics(fov_deg, device=device)
            camera_params = torch.cat(
                [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
            )
            conditioning_params = torch.cat(
                [
                    conditioning_cam2world_pose.reshape(-1, 16),
                    intrinsics.reshape(-1, 9),
                ],
                1,
            )

            ws = G.mapping(
                z,
                conditioning_params,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
            )
            img = G.synthesis(ws, camera_params)["image"]
            imgs.append(
                torch.nn.functional.interpolate(
                    img, size=(64, 64), mode="bilinear", align_corners=False
                )
            )

            img = (
                (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
            )
            PIL.Image.fromarray(img.cpu().numpy(), "RGB").save(
                f"{result_dir}/frame_{frame_idx:05d}.png"
            )

        imgs = torch.cat(imgs)
        save_image(
            imgs, f"{result_dir}/all.png", nrow=12, normalize=True, scale_each=True
        )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    generate_images(outdir="results")  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
