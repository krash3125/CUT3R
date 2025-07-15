import cv2
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import time
import numpy as np
import os


def ssim_compare(img1_path, img2_path, dim=(112, 112)):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("One or both image paths are invalid or unreadable.")

    if img1.shape != dim:
        img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    if img2.shape != dim:
        img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

    return ssim(img1, img2)


def ssim_compare_frames(frame1, frame2, dim=(112, 112)):
    try:
        if frame1.shape != dim:
            img1 = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
        if frame2.shape != dim:
            img2 = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2)
    except Exception as e:
        print(f"Error comparing frames: {e}")
        return 0
