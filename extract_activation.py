# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models_hook import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
from time import time
from torchvision.transforms import v2
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
SAVE_PATH = "/home/yifulu/work/sae/SiT_IN10_20lay_t0.1"

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
def main( args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = False

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    model=torch.compile(model,fullgraph =True,mode="max-autotune")
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    # transport_sampler = Sampler(transport)
    
    print("created sampler")
    
    transform = v2.Compose([
        v2.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),  
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        prefatch_factor=8
        )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    print("load vae")
    timestep = 0.1
    i = 0
    buffer = []
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            t = torch.full(y.shape,timestep,device=device)
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            t, xt, ut = transport.path_sampler.plan(t, torch.randn_like(x), x)
            model_output = model(xt, t, **model_kwargs)
            buffer.append(model_output.cpu())
            i+=1
            if i%5 == 0:
                tens = torch.cat(buffer)
                torch.save(tens, os.path.join(SAVE_PATH,f"output_{i//5}.pt"))
                buffer = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
   
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")

    parser.add_argument("--batch-size", type=int, default=256)

    parse_transport_args(parser)
   
    args = parser.parse_known_args()[0]
    main( args)
