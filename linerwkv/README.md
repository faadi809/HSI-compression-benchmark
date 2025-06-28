# LineRWKV

Code for ["Onboard deep lossless and near-lossless predictive coding of hyperspectral images with line-based attention"](https://arxiv.org/abs/2403.17677) paper.

LineRWKV is a method for lossless and lossy compression of hyperspectral images. The compression algorithm is based on predictive coding where a neural network performs prediction of each pixel based on a causal spatial and spectral context, followed by entropy coding of the prediction residual. The neural network predictor processes the image line-by-line using a novel hybrid attentive-recursive operation that combines the representational advantages of Transformers with the linear complexity and recursive implementation of recurrent neural networks. This allows significant savings in memory and computational complexity while reaching state-of-the-art rate-distortion performance.

BibTex reference:
```
@article{valsesia2024linerwkv,
  title={Onboard deep lossless and near-lossless predictive coding of hyperspectral images with line-based attention},
  author={Valsesia, Diego and Bianchi, Tiziano and Magli, Enrico},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  year={2024}
}
```

## Dataset
The code is reconfigured for PaviaU dataset.


## Model configuration
Model hyperparameters can be changed via the config.py file. 
```

## Training
Launch the training script
```
./launcher_train.sh
```

## Testing/Compressing an image
**Compression tests require image files to be provided as raw files with BSQ ordering, little-endian 16-bit unsigned integers.**
Compress a BSQ raw image with
```
./launcher_compress.sh
```
The quantization step size is set via parameter delta as 2*delta+1. delta=0 performs lossless compression. Paper results use delta=(0 1 2 3 4 5 10 20 40).

The compressed files are the script variables:
- $compressed_file : this contains the entropy-encoded residuals
- $numerical_warning_file : this contains compressed signaling to prevent numerical errors
- $side_info_file_{mu,sigma} : this contains side information about normalizations for decoding

Entropy encoder and decoder are provided as binary files, compiled for Linux x86-64. They are directly called by the compressor.py and decompressor.py files.
