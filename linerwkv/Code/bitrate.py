import os
import math

# Get size of all compressed components (in bytes)
cmp_size = os.path.getsize("/home/data/Fahad/codes/codes_for_Review/linerwkv-main/Results/linerwkv/20250614_1423/PaviaU_p_compressed_0.cmp")
mu_size = os.path.getsize("/home/data/Fahad/codes/codes_for_Review/linerwkv-main/Results/linerwkv/20250614_1423/PaviaU_p_side_info_0.h5_mu.cmp")
sigma_size = os.path.getsize("/home/data/Fahad/codes/codes_for_Review/linerwkv-main/Results/linerwkv/20250614_1423/PaviaU_p_side_info_0.h5_sigma.cmp")

total_bytes = cmp_size + mu_size + sigma_size

# Convert to bits
total_bits = total_bytes * 8

# PaviaU: width=610, height=340, bands=103
pixels = 610 * 340
bands = 103
denominator = pixels * bands

# Bitrate in bits per pixel per channel
bitrate = total_bits / denominator
print(f"Bitrate (bpppc): {bitrate:.6f}")
