# HyCoT: Hyperspectral Compression Transformer with an Efficient Training
This repository contains code of the paper [`HyCoT: Hyperspectral Compression Transformer with an Efficient Training`](https://arxiv.org/abs/2408.08700) submitted to WHISPERS 2024 . This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) by [Martin Hermann Paul Fuchs](https://rsim.berlin/team/members/martin-hermann-paul-fuchs), [Behnood Rasti](https://rsim.berlin/team/members/behnood-rasti) and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

If you use this code, please cite our paper given below:

> M. H. P. Fuchs, B. Rasti and B. Demіr, "[HyCoT: Hyperspectral Compression Transformer with an Efficient Training](https://arxiv.org/abs/2408.08700)", 14th IEEE GRSS Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS), 2024, under review.

```
@article{Fuchs:2024,
    author={M. H. P. {Fuchs}, B. {Rasti} and B. {Demіr}},
    journal={14th IEEE GRSS Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS)}, 
    title={HyCoT: Hyperspectral Compression Transformer with an Efficient Training Strategy}, 
    year={2024, under review}
}
```
This repository contains code that has been adapted from the CompressAI [\[2\]](#2-compressai) framework https://github.com/InterDigitalInc/CompressAI/.

## Description
The development of learning-based hyperspectral image (HSI) compression models has recently attracted significant interest. Existing models predominantly utilize convolutional filters, which capture only local dependencies. Furthermore, they often incur high training costs and exhibit substantial computational complexity. To address these limitations, in this paper we propose Hyperspectral Compression Transformer (HyCoT) that is a transformer-based autoencoder for pixelwise HSI compression. Additionally, we introduce an efficient training strategy to accelerate the training process. Experimental results on the HySpecNet-11k dataset demonstrate that HyCoT surpasses the state-of-the-art across various compression ratios by over 1 dB with significantly reduced computational requirements. Our code and pre-trained weights are publicly available at https://git.tu-berlin.de/rsim/hycot.

## Setup
The code in this repository is tested with `Ubuntu 22.04 LTS` and `Python 3.10.6`.

### Dependencies
All dependencies are listed in the [`requirements.txt`](requirements.txt) and can be installed via the following command:
```
pip install -r requirements.txt
```

### Dataset
HySpecNet-11k is made up of image patches acquired by the Environmental Mapping and Analysis Program (EnMAP) [\[1\]](#1-environmental-mapping-and-analysis-program-enmap) satellite.

Follow the instructions on [https://hyspecnet.rsim.berlin](https://hyspecnet.rsim.berlin) to download, extract and preprocess the HySpecNet-11k dataset.

The folder structure should be as follows:
```
┗ 📂 hycot/
  ┗ 📂 datasets/
    ┗ 📂 hyspecnet-11k/
      ┣ 📂 patches/
      ┃ ┣ 📂 tile_001/
      ┃ ┃ ┣ 📂 tile_001-patch_01/
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-DATA.npy
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_PIXELMASK.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CIRRUS.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CLASSES.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CLOUD.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CLOUDSHADOW.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_HAZE.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_SNOW.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_TESTFLAGS.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_SWIR.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_VNIR.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-SPECTRAL_IMAGE.TIF
      ┃ ┃ ┃ ┗ 📜 tile_001-patch_01-THUMBNAIL.jpg
      ┃ ┃ ┣ 📂 tile_001-patch_02/
      ┃ ┃ ┃ ┗ 📜 ...
      ┃ ┃ ┗ 📂 ...
      ┃ ┣ 📂 tile_002/
      ┃ ┃ ┗ 📂 ...
      ┃ ┗ 📂 ...
      ┗ 📂 splits/
        ┣ 📂 easy/
        ┃ ┣ 📜 test.csv
        ┃ ┣ 📜 train.csv
        ┃ ┗ 📜 val.csv
        ┣ 📂 hard/
        ┃ ┣ 📜 test.csv
        ┃ ┣ 📜 train.csv
        ┃ ┗ 📜 val.csv
        ┗ 📂 ...
```

## Usage

### Train
The [`train.py`](train.py) expects the following command line arguments:
| Parameter | Description | Default |
| :- | :- | :- |
| `--devices` | Devices to use, e.g. `cpu` or `0` or `0,2,5,7` | `0` |
| `--train-batch-size` | Training batch size | `2` |
| `--val-batch-size` | Validation batch size | `2` |
| `-n` | Data loaders threads | `4` |
| `-d` | Path to dataset | `./datasets/hyspecnet-11k/` |
| `--mode` | Dataset split difficulty | `easy` |
| `--transform` | Dataset transformation, e.g. `random_16x16` | `None` |
| `-m` | Model architecture | `hycot_cr4` |
| `--loss` | Loss | `mse` |
| `-e` | Number of epochs | `2000` |
| `-lr` | Learning rate | `1e-3` |
| `--save-dir` | Directory to save results | `./results/trains/` |
| `--seed` | Set random seed for reproducibility | `10587` |
| `--clip-max-norm` | Gradient clipping max norm | `1.0` |
| `--checkpoint` | Path to a checkpoint to resume training | `None` |

Specify the parameters in the [`train.sh`](train.sh) file and then execute the following command:
```console
./train.sh
```
Or run the python code directly through the console:
```console
python train.py \
    --devices 0 \
    --train-batch-size 2 \
    --val-batch-size 2 \
    --num-workers 4 \
    --learning-rate 1e-3 \
    --mode easy \
    --model hycot_cr4 \
    --loss mse \
    --epochs 2000
```
### Test
The [`test.py`](test.py) expects the following command line arguments:
| Parameter | Description | Default |
| :- | :- | :- |
| `--device` | Device to use (default: 0), e.g. `cpu` or `0` | `0` |
| `--batch-size` | Test batch size | `4` |
| `-n` | Data loaders threads | `0` |
| `-d` | Path to dataset | `./datasets/hyspecnet-11k/` |
| `--mode` | Dataset split difficulty | `easy` |
| `-m` | Model architecture | `hycot_cr4` |
| `--checkpoint` | Path to the checkpoint to evaluate | `None` |
| `--half` | Convert model to half floating point (fp16) | `False` |
| `--save-dir` | Directory to save results | `./results/tests/` |
| `--seed` | Set random seed for reproducibility | `10587` |

Specify the parameters in the [`test.sh`](test.sh) file and then execute the following command:
```console
./test.sh
```
Or run the python code directly through the console:
```console
python test.py \
    --device 0 \
    --batch-size 4 \
    --num-workers 4 \
    --mode easy \
    --model hycot_cr4 \
    --checkpoint ./results/weights/hycot_cr4.pth.tar
```

## Pre-Trained Weights
Pre-trained weights are publicly available and should be downloaded into the [`./results/weights/`](results/weights/) folder.

| Method | Model | Compression Ratio | PSNR | Download Link |
| :----- | :---- | :--- | :--- | :------------ |
| 1D-CAE [\[3\]](#3-1d-convolutional-autoencoder-1d-cae) | `cae1d_cr32` | 28.86 | 48.95 dB | [cae1d_1bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/ew2jr67yro7cj3x/download/cae1d_1bpppc.pth.tar) |
| | `cae1d_cr16` | 15.54 | 52.38 dB | [cae1d_2bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/Ae35EBRado8QSmk/download/cae1d_2bpppc.pth.tar) |
| | `cae1d_cr8` | 7.77 | 53.90 dB | [cae1d_4bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/ZNeXycsssRdYZ5m/download/cae1d_4bpppc.pth.tar) |
| | `cae1d_cr4` | 3.96 | 54.85 dB | [cae1d_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/GpmXDAWEeo2nG5w/download/cae1d_8bpppc.pth.tar) |
| SSCNet [\[4\]](#4-spectral-signals-compressor-network-sscnet) | `sscnet_cr32` | 32.00 | 43.24 dB | [sscnet_1bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/wPwbMKYJAmXxLRX/download/sscnet_1bpppc.pth.tar) |
| | `sscnet_cr16` | 15.84 | 43.60 dB | [sscnet_2bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/H9Yg8n8rzxGMe2Z/download/sscnet_2bpppc.pth.tar) |
| | `sscnet_cr8` | 8.00 | 43.69 dB | [sscnet_4bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/WQ65aCDxgedQYxZ/download/sscnet_4bpppc.pth.tar) |
| | `sscnet_cr4` | 3.96 | 43.29 dB | [sscnet_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/5kiQ8ZLRnkpbSg6/download/sscnet_8bpppc.pth.tar) |
| 3D-CAE [\[5\]](#5-3d-convolutional-auto-encoder-3d-cae) | `cae3d_cr32` | 31.69 | 39.06 dB | [cae3d_1bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/QDfARfWL3Pab3xK/download/cae3d_1bpppc.pth.tar) |
| | `cae3d_cr16` | 15.84 | 39.54 dB | [cae3d_2bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/dD3qtjrgzJxmymP/download/cae3d_2bpppc.pth.tar) |
| | `cae3d_cr8` | 7.92 | 39.69 dB | [cae3d_4bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/CmTdQzcE3x9pEEJ/download/cae3d_4bpppc.pth.tar) |
| | `cae3d_cr4` | 3.96 | 39.94 dB | [cae3d_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/DpqKJdMbojF3CLx/download/cae3d_8bpppc.pth.tar) |
| HyCoT | `hycot_cr32` | 28.86 | 50.26 dB | [hycot_cr32.pth.tar](https://tubcloud.tu-berlin.de/s/QsT3An5WTPbDXQS/download/hycot_cr32.pth.tar) |
| | `hycot_cr16` | 15.54 | 53.20 dB | [hycot_cr16.pth.tar](https://tubcloud.tu-berlin.de/s/5jGeG29kTJfHX58/download/hycot_cr16.pth.tar) |
| | `hycot_cr8` | 7.77 | 55.38 dB | [hycot_cr8.pth.tar](https://tubcloud.tu-berlin.de/s/As8yaM3k2isjX92/download/hycot_cr8.pth.tar) |
| | `hycot_cr4` | 3.96 | 56.29 dB | [hycot_cr4.pth.tar](https://tubcloud.tu-berlin.de/s/jeaSXYQHN7ki3mE/download/hycot_cr4.pth.tar) |

## Authors
**Martin Hermann Paul Fuchs**
https://rsim.berlin/team/members/martin-hermann-paul-fuchs

## License
The code in this repository is licensed under the **MIT License**:
```
MIT License

Copyright (c) 2024 Martin Hermann Paul Fuchs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## References
### [1] [Environmental Mapping and Analysis Program (EnMAP)](https://doi.org/10.3390/rs70708830)

### [2] [CompressAI](https://doi.org/10.48550/arXiv.2011.03029)

### [3] [1D-Convolutional Autoencoder (1D-CAE)](https://doi.org/10.5194/isprs-archives-XLIII-B1-2021-15-2021)

### [4] [Spectral Signals Compressor Network (SSCNet)](https://doi.org/10.3390/rs14102472)

### [5] [3D Convolutional Auto-Encoder (3D-CAE)](https://doi.org/10.1117/1.JEI.30.4.041403)