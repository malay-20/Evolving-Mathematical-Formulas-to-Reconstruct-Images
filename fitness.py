import numpy as np
from skimage.metrics import structural_similarity as ssim

def get_score(target, generated, formula_size):
    mse = np.mean((target - generated) ** 2)
    penalty = formula_size * 0.02
    return -mse - penalty

def get_metrics(target, generated):
    mse = np.mean((target - generated) ** 2)
    try:
        ssim_val = ssim(target, generated, data_range=255)
    except:
        ssim_val = 0.0
        
    return {'MSE': mse, 'SSIM': ssim_val}