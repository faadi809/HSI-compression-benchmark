Note: This code is taken from https://git.tu-berlin.de/rsim/hycot where it was uploaded by the authors of  " HyCoT: Hyperspectral Compression Transformer with an Efficient Training" which is adapted from the CompressAI [2] framework https://github.com/InterDigitalInc/CompressAI/.


If you use this code, please cite the paper given below:

> M. H. P. Fuchs, B. Rasti and B. Demіr, "[HyCoT: Hyperspectral Compression Transformer with an Efficient Training](https://arxiv.org/abs/2408.08700)", 14th IEEE GRSS Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS), 2024, under review.

```
@article{Fuchs:2024,
    author={M. H. P. {Fuchs}, B. {Rasti} and B. {Demіr}},
    journal={14th IEEE GRSS Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS)}, 
    title={HyCoT: Hyperspectral Compression Transformer with an Efficient Training Strategy}, 
    year={2024, under review}
}
```
The code for 4 models is implemented in this repository, namely: 1dCAE [3], 2dCAE (SSCNet) [4], 3dCAE [5], and HyCoT [6].



### Dependencies
All dependencies are listed in the [`requirements.txt`](requirements.txt) and can be installed via the following command:
```
pip install -r requirements.txt
```

### Dataset
The code is adapted for PaviaU dataset. The path of paviaU.mat must be provided in the paviau.py file present in datasets directory.

## Usage

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

The model labels for 1dCAE, SSCNet, 3dCAE and HycoT at different bitrates can be taken from the __init()__.py in models diectory.
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



## References
### [1] [Environmental Mapping and Analysis Program (EnMAP)](https://doi.org/10.3390/rs70708830)

### [2] [CompressAI](https://doi.org/10.48550/arXiv.2011.03029)

### [3] [1D-Convolutional Autoencoder (1D-CAE)](https://doi.org/10.5194/isprs-archives-XLIII-B1-2021-15-2021)

### [4] [Spectral Signals Compressor Network (SSCNet)](https://doi.org/10.3390/rs14102472)

### [5] [3D Convolutional Auto-Encoder (3D-CAE)](https://doi.org/10.1117/1.JEI.30.4.041403)
### [6] [HyCoT: Hyperspectral Compression Transformer with an Efficient Training](https://arxiv.org/abs/2408.08700)"
