# HINER: Neural Representation for Hyperspectral Image

* **Note:** This HINER repository is taken from https://paperswithcode.com/paper/hiner-neural-representation-for-hyperspectral#code. No changes were made to this code as it was already configured for PaviaU dataset which aligns with our objective. We are thankful to the authors for sharing their code. The sole purpose of presenting this code here is to facilitate the readers and developers and make access easier to all the codes discussed in our paper.
  
## Overview
This is the official implementation of HINER (ACM MM'24), a novel neural representation for **compressing HSI** and **ensuring high-quality downstream tasks on compressed HSI**.

* **Compressing HSI:** HINER fully exploits inter-spectral correlations by **explicitly encoding of spectral wavelengths** and achieves a compact representation of the input HSI sample through joint optimization with a learnable decoder. By additionally incorporating the **Content Angle Mapper** with the L1 loss, we can supervise the global and local information within each spectral band, thereby enhancing the overall reconstruction quality.

* **Ensuring high-quality downstream tasks on compressed HSI:** For downstream classification on compressed HSI, we theoretically demonstrate the task accuracy is not only related to the classification loss but also to the reconstruction fidelity through a first-order expansion of the accuracy degradation, and accordingly adapt the reconstruction by introducing **Adaptive Spectral Weighting**. Owing to the monotonic mapping of HINER between wavelengths and spectral bands, we propose **Implicit Spectral Interpolation** for data augmentation by adding random variables to input wavelengths during classification model training.

* **Experiments:** Experimental results on various HSI datasets demonstrate the superior compression performance of our HINER compared to the existing learned methods and also the traditional codecs. Our model is lightweight and computationally efficient, which maintains high accuracy for downstream classification task even on decoded HSIs at high compression ratios.


## News
✅ 2024.12: HSI Compression has been incorporated

✅ 2025.02: Classification on compressed HSI has been incorporated

✅ 2025.02: More implementations for compared literatures can be used.



## Requirement
```bash
pip install -r requirements.txt
```

## Quick Usage for Compression
1. **Train HINER in IndianPine**
```bash
CUDA_VISIBLE_DEVICES=0 python train_hiner.py  --outf HINER \
    --data_path data/IndianPine.mat --vid IndianPine.mat --data_type HSI \
    --arch hiner --conv_type none pshuffel --act gelu --norm none  --crop_list 180_180  --ori_shape 146_146_200 \
    --resize_list -1 --loss CAM  --enc_dim 64_16 \
    --quant_model_bit 8 --quant_embed_bit 8 --fc_hw 3_3 \
    --dec_strds 5 3 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.0  -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.001
```

2. **Train HINER in PaviaUniversity**
```bash
CUDA_VISIBLE_DEVICES=1 python train_hiner.py  --outf HINER \
    --data_path data/PaviaU.mat --vid PaviaU.mat --data_type HSI  \
    --arch hiner --conv_type convnext pshuffel --act gelu --norm none  --crop_list 720_360  --ori_shape 610_340_103 \
    --resize_list -1 --loss CAM  --enc_dim 64_16 \
    --quant_model_bit 8 --quant_embed_bit 8 --fc_hw 6_3\
    --dec_strds 5 4 3 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.0  -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.001
```

* You can change **--modelsize** for different bitrate
* More epoch means better performance while increasing encoding time
* You can change hyperparameters for various architecture and optimization strategies.


## Quick Usage for Classification on Compressed HSI
1. **Train HINER-Classification in IndianPine with SpectralFormer**
```bash
CUDA_VISIBLE_DEVICES=2 python train_hiner_cls.py  --outf HINER_CLS \
    --data_path data/IndianPine.mat  --vid IndianPine.mat --data_type HSI  \
    --arch hiner --conv_type convnext pshuffel --act gelu --norm none  --crop_list 180_180  \
    --resize_list -1 --loss CAM  --enc_dim 64_16 --fc_hw 3_3  \
    --quant_model_bit 8 --quant_embed_bit 8  \
    --dec_strds 5 3 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 0.5  -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.001   \
    --patches=7 --band_patches=3 --mode='CAF' --weight_decay=5e-3  \
    --weight your_path
```

* You can change **--modelsize** for different bitrate
* You can change hyperparameters for various architecture and optimization strategies.


## Implementations for Compared Literatures
You can change **--arch** and **--mode** to test other networks. For reproducing other methods, you should keep naive hyperparameters as their paper.

## Contact
Junqi Shi: junqishi@smail.nju.edu.cn

## Citation
If this work assists your research, feel free to give a star ⭐ or cite using:
```bash
@inproceedings{shi2024hiner,
  title={HINER: Neural Representation for Hyperspectral Image},
  author={Shi, Junqi and Jiang, Mingyi and Lu, Ming and Chen, Tong and Cao, Xun and Ma, Zhan},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={9837--9846},
  year={2024}
}
```

## Acknowledgement
This framework is based on [HNeRV](https://github.com/haochen-rye/HNeRV) and [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer)

We thank the authors for sharing their codes.
