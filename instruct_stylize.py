from __future__ import annotations

import math
import random
import shutil
import sys
import os
from argparse import ArgumentParser

from tqdm import tqdm, trange

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

sys.path.append("./ip2p")
for root, dirs, files in os.walk("./ip2p"):
    for dir in dirs:
        sys.path.append(os.path.join(root, dir))

from ip2p.stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [
                torch.cat(
                    [
                        cond["c_crossattn"][0],
                        uncond["c_crossattn"][0],
                        uncond["c_crossattn"][0],
                    ]
                )
            ],
            "c_concat": [
                torch.cat(
                    [cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]]
                )
            ],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(
            cfg_z, cfg_sigma, cond=cfg_cond
        ).chunk(3)
        return (
            out_uncond
            + text_cfg_scale * (out_cond - out_img_cond)
            + image_cfg_scale * (out_img_cond - out_uncond)
        )


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: (
                vae_sd[k[len("first_stage_model.") :]]
                if k.startswith("first_stage_model.")
                else v
            )
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="ip2p/configs/generate.yaml", type=str)
    parser.add_argument(
        "--ckpt", default="pretrained_models/instruct-pix2pix-00-22000.ckpt", type=str
    )
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--eg3dseed", default=1234, type=int)
    parser.add_argument("--outdir", default="dataset", type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    dataset_seed = f"seed{args.eg3dseed:04d}"
    multiviewdir = os.path.join(args.outdir, dataset_seed, "multiview")
    stylizeddir = os.path.join(args.outdir, dataset_seed, "stylized")

    assert os.path.exists(multiviewdir)
    assert os.path.isdir(multiviewdir)

    if os.path.exists(stylizeddir):
        shutil.rmtree(stylizeddir)
    os.makedirs(stylizeddir, exist_ok=True)

    if args.edit:
        with open(os.path.join(args.outdir, dataset_seed, "edit.txt"), "w") as f:
            f.write(args.edit)

    if args.test:
        input_imgs = [os.path.join(multiviewdir, "frame_00042.png")]
    else:
        input_imgs = []
        for frame in os.listdir(multiviewdir):
            if frame == "all.png":
                continue
            # print(os.path.join(multiviewdir, frame), " -> ", os.path.join(stylizeddir, frame))
            input_imgs.append(os.path.join(multiviewdir, frame))

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    for img_path in tqdm(input_imgs):
        iter_num = 20 if not args.test else 1
        img_name = img_path.split("/")[-1].split(".")[0]

        output_path = [
            os.path.join(stylizeddir, f"{i+1:02d}_{img_name}.png")
            for i in range(iter_num)
        ]

        for idx in trange(iter_num):
            seed = random.randint(0, 100000) if args.seed is None else args.seed
            input_image = Image.open(img_path).convert("RGB")
            width, height = input_image.size
            factor = args.resolution / max(width, height)
            factor = (
                math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            )
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(
                input_image, (width, height), method=Image.Resampling.LANCZOS
            )

            if args.edit == "":
                input_image.save(output_path[idx])
                return

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
                input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_image = rearrange(input_image, "h w c -> 1 c h w").to(
                    model.device
                )
                cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                sigmas = model_wrap.get_sigmas(args.steps)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": args.cfg_text,
                    "image_cfg_scale": args.cfg_image,
                }
                torch.manual_seed(seed)
                z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(
                    model_wrap_cfg, z, sigmas, extra_args=extra_args
                )
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_image.save(output_path[idx])


# python instruct_stylize.py --edit "turn it into the Hulk" --test
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    main()
