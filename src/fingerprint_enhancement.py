#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Utility functions for fingerprint image enhancement"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import cv2
from skimage.morphology import thin


def otsu(image: np.ndarray, gaussian_blur_size: int = 5) -> np.ndarray:
    """
    Otsu binarization
    :param image: the original image
    :param gaussian_blur_size: the size of the Gaussian kernel
    :return: the binarized image
    """
    blur = cv2.GaussianBlur(image, (gaussian_blur_size, gaussian_blur_size), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3/255


def ridge_thinning(image: np.ndarray) -> np.ndarray:
    """
    Ridge thinning or image skeletonization
    :param image: the original image
    :return: the skeleton of the image
    """
    thinned = thin(image)
    thinned = thinned.astype(np.float)
    return thinned


def skeleton_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Perform skeleton enhancement performing a sequence of morphological operations
    :param image: the original skeletonized image
    :return: an improved version of the skeletonized image
    """
    tmp = clean(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel, iterations=1)
    enhanced_skeleton = ridge_thinning(closing)


def clean(image: np.ndarray) -> np.ndarray:
    """
    Perform the clean morphological operation
    :param image: the original image
    :return: the cleaned image
    """
    height, width = image.shape
    cleaned_img = np.zeros(image.shape, image.dtype)

    for i in range(height):
        for j in range(width):
            cleaned_img[i][j] = image[i][j]
            if cleaned_img[i][j] == 1:
                count = 0
                for h in range(max(0, i-2), min(height, i+3)):
                    for k in range(max(0, j-2), min(width, j+3)):
                        if image[h][k] == 1:
                            count += 1
                if count <= 2:
                    cleaned_img[i][j] = 0
    return cleaned_img


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Image normalization
    :param image: the image to normalize
    :return: the normalized image
    """
    image = image.astype(float)
    image = np.copy(image)
    image -= np.min(image)
    m = np.max(image)
    if m > 0.0:
        image *= 1.0/m
    return image

#
# def local_normalize(image: np.ndarray, block_size: int = 32) -> np.ndarray:
#     """
#     Boh!
#     :param image:
#     :param block_size:
#     :return:
#     """
#     image = np.copy(image)
#     height, width = image.shape
#     for y in range(0, height, block_size):
#         for x in range(0, width, block_size):
#             image[y:y+block_size, x:x+block_size] = normalize(image[y:y+block_size, x:x+block_size])
#     return image


#TODO find good values for m0 and v0
def normalize2(image: np.ndarray, m0: int, v0: int) -> np.ndarray:
    new = np.zeros(image.shape, dtype=image.dtype)
    height, width = image.shape
    mean = np.mean(image)
    var = np.var(image)
    for i in range(height):
        for j in range(width):
            if image[i][j] > mean:
                new[i][j] = m0 + np.sqrt(((image[i][j] - mean)**2)*v0/var)
            else:
                new[i][j] = m0 - np.sqrt(((image[i][j] - mean)**2)*v0/var)
    return new


def roi_mask(image: np.ndarray, threshold: float = 0.1, block_size: int = 32) -> np.ndarray:
    """
    Create a mask image consisting of only 0's and 1's.
    The areas containing 1's represent the areas that look interesting to us,
    meaning that they contain a good variety of color values
    :param image: the original image
    :param threshold: the standard deviation threshold
    :param block_size: the window size
    :return: an image mask containing the ROI
    """
    mask = np.empty(image.shape)
    height, width = image.shape

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            standardDeviation = np.std(block)

            if standardDeviation < threshold:
                mask[y:y+block_size, x:x+block_size] = 0.0
            elif block.shape != (block_size, block_size):
                mask[y:y+block_size, x:x+block_size] = 0.0
            else:
                mask[y:y+block_size, x:x+block_size] = 1.0
    return mask


def binarization(img: np.ndarray, block_size: int = 32, mean_threshold_factor: float = 1.2) -> np.ndarray:
    """
    Image binarization using local block analysis
    :param img: the original image
    :param block_size: the window size
    :param mean_threshold_factor: mean block threshold factor
    :return: the binarized image
    """
    height, width = img.shape
    new = np.zeros(img.shape)
    i = j = 0

    while i < height:
        while j < width:
            img_block = np.zeros((block_size, block_size), dtype=img.dtype)
            count = 0
            pixel_sum = 0
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    img_block[h-i][k-j] = img[h][k]
                    count += 1
                    pixel_sum += img[h][k]
            mean = pixel_sum/count

            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    if img_block[h-i][k-j] > mean_threshold_factor*mean:
                        new[h][k] = 1
                    else:
                        new[h][k] = 0
            j += block_size
        i += block_size
        j = 0
    return new
