import numpy as np
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
#from spectral.algorithms import sam  # Correct import for SAM

# --- Load Original (BSQ) ---
orig = np.fromfile('/home/data/Fahad/codes/codes_for_Review/linerwkv-main/PaviaU_p.bsq', dtype=np.uint16).reshape((624, 352, 103))  # Adjust shape!

# --- Load Reconstructed (.mat) ---
recon = sio.loadmat("/home/data/Fahad/codes/codes_for_Review/linerwkv-main/Results/linerwkv/20250614_1423/PaviaU_p_reconstructed_0.mat")["data_rec"]
print(recon.shape)

# --- Check Shapes ---
assert orig.shape == recon.shape, f"Shape mismatch: {orig.shape} vs {recon.shape}"

# --- Normalize ---
if orig.max() > 1:
    orig = orig / 65535.0
    recon = recon / 65535.0

# --- PSNR/SSIM ---
psnr_vals, ssim_vals = [], []
for b in range(orig.shape[2]):
    psnr_vals.append(psnr(orig[:, :, b], recon[:, :, b], data_range=1.0))
    ssim_vals.append(ssim(orig[:, :, b], recon[:, :, b], data_range=1.0))

avg_psnr = np.mean(psnr_vals)
avg_ssim = np.mean(ssim_vals)

def sam(x, y):
    """Manual SAM implementation."""
    dot_product = np.sum(x * y, axis=1)
    norm_x = np.linalg.norm(x, axis=1)
    norm_y = np.linalg.norm(y, axis=1)
    cos_theta = dot_product / (norm_x * norm_y + 1e-10)
    return np.arccos(cos_theta)

# --- SAM ---
orig_reshaped = orig.reshape(-1, orig.shape[2])
recon_reshaped = recon.reshape(-1, recon.shape[2])
sam_vals = sam(orig_reshaped, recon_reshaped)
avg_sam = np.mean(sam_vals)


print(f"PSNR: {avg_psnr:.2f} dB")
print(f"SSIM: {avg_ssim:.4f}")
print(f"SAM: {avg_sam:.4f} radians")
