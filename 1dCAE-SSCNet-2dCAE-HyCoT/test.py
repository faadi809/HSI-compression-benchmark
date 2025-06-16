# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import json
import random
import sys
import time
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

#from datasets.hyspecnet11k import HySpecNet11k
from datasets.paviau import PaviaUDataset
from models import models
from metrics import metrics
from utils import checkpoint

# Setup

 

@torch.no_grad()
def inference(model, x):
    start = time.time()
    y = model.compress(x)
    enc_time = time.time() - start

    start = time.time()
    x_hat = model.decompress(y)
    dec_time = time.time() - start

    bpppc = model.bpppc
    cr = model.compression_ratio
    psnr = metrics["psnr"]()(x, x_hat)
    ssim = metrics["ssim"]()(x, x_hat)
    sa = metrics["sa"]()(x, x_hat)

    return {
        "bpppc": bpppc,
        "cr": cr,
        "psnr": psnr.item(),
        "ssim": ssim.item(),
        "sa": sa.item(),
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


def eval_model(model, test_dataloader, half=False):
    device = next(model.parameters()).device
    metrics = defaultdict(float)

    loop = tqdm(test_dataloader, leave=True)
    loop.set_description(f"Testing")
    for x in loop:
        x = x.to(device)
        if half:
            model = model.half()
            x = x.half()
        rv = inference(model, x)
        for k, v in rv.items():
            metrics[k] += v

    for k, v in metrics.items():
        metrics[k] = v / len(test_dataloader)

    return metrics


def main(argv):
    args = parse_args(argv)

    # improve reproducibility
    torch.backends.cudnn.deterministic = True
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    torch.set_num_threads(1)

    results = defaultdict(list)

    model = models[args.model](src_channels=args.num_channels)

    device = f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    model = model.to(device)

    checkpoint.load_checkpoint_eval(args.checkpoint, model)

   # test_dataset = HySpecNet11k(args.dataset, mode=args.mode, split="test", transform=None)
    test_dataset = PaviaUDataset(args.dataset, split="test", transform=None)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    metrics = eval_model(model, test_dataloader, args.half)

    for k, v in metrics.items():
        results[k].append(v)

    output = {
        "name": args.model,
        "description": f"Test",
        "results": results,
    }

    save_file = f'{args.save_dir}/{args.checkpoint.split("/")[-2]}.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, 'w') as file:
        json.dump(output, file, indent=2)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Test script.")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device to use (default: %(default)s), e.g. cpu or 0"
    )
    parser.add_argument(
        "--batch-size",
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Data loaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="./home/data/Fahad/HSI datasets/PaviaU/",
        help="Path to dataset (default: %(default)s)"
    )
    
    parser.add_argument(
        "--num-channels",
        type=int,
        default=103,
        help="Number of data channels, (default: %(default)s)"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="cae1d_cr114",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/data/Fahad/results/paviau/2025_06_10-09_04_01-cae1d_cr114-mse-0.001/best.pth.tar",
        help="Path to the checkpoint to evaluate"
    )
    parser.add_argument(
        "--half",
        default=False,
        action="store_true",
        help="Convert model to half floating point (fp16)",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="./results/tests/",
        help="Directory to save results (default: %(default)s)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set random seed for reproducibility (default: %(default)s)"
    )

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(sys.argv[1:])
