from metrics import mse
from metrics import psnr
from metrics import ssim
from metrics import sa

metrics = {
    "mse": mse.MeanSquaredError,
    "mae": mse.MeanSquaredError,
    "psnr": psnr.PeakSignalToNoiseRatio,
    "ssim":ssim.StructuralSimilarity,
    "sa": sa.SpectralAngle,
}
