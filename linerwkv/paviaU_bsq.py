import scipy.io
import numpy as np

# Load the .mat file
mat = scipy.io.loadmat('/home/data/Fahad/HSI_datasets/PaviaU/PaviaU.mat')
hsi = mat['paviaU']  # replace with your actual variable name

# Confirm shape is (H, W, 103)
print('Shape:', hsi.shape)


# Convert to BSQ format: (Bands, Height, Width)
hsi_bsq = np.transpose(hsi, (2, 0, 1)).astype(np.uint16)

# Pad to [103, 624, 352]
#pad_h = 624 - hsi_bsq.shape[1]  # 14
#pad_w = 352 - hsi_bsq.shape[2]  # 12
data_padded = np.pad(hsi_bsq, ((0, 0), (0, 14), (0, 12)), mode='reflect')
print("data padded:", data_padded.shape)
# Save to .bsq file
data_padded.tofile('/home/data/Fahad/codes/codes_for_Review/linerwkv-main/PaviaU.bsq')
print("BSQ file saved")
