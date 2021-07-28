import numpy as np
import torch
import cv2 as cv

# FGSM attack code


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def gaussian_attack(image, epsilon=0.5):
    noise = np.random.normal(0, epsilon, image.shape)
    out = image + noise
    return out


def light_attack(image, alpha=1.0, beta=20.0):
    return np.clip(cv.convertScaleAbs(image, alpha=alpha, beta=beta), 0, 255)
