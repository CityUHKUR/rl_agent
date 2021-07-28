import numpy as np
from skimage import io, filters, img_as_float
from skimage.metrics import structural_similarity 
from skimage.metrics import mean_squared_error 


def ssim(ref, test_img):
    return structural_similarity(ref, test_img)


def mse(ref, test_img):
    return mean_squared_error(ref, test_img)
