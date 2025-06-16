from losses import mseloss
from losses import maeloss
from losses import saloss
#from losses import mixedmsesaloss
#from losses import mixedmsesassimloss
#from losses import rdloss

losses = {
    "mse": mseloss.MeanSquaredErrorLoss,
    "mae": maeloss.MeanAbsoluteErrorLoss,
    "sa" : saloss.SpectralAngleLoss,
   # "mixed_mse_sa": mixedmsesaloss.MixedMseSaLoss,
    #"mixed_mse_sa_ssim": mixedmsesassimloss.MixedMseSaSsimLoss,
    #"rdloss": rdloss.RateDistortionLoss,
}
