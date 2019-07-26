#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Main module"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import logging
from skimage.morphology import skeletonize
import cv2

from utils import load_image
from utils import neighbor_coordinates
from utils import display_image
from utils import print_images
from utils import print_images_args


def fft_enhancement(img: np.ndarray) -> np.ndarray:
    """
    Image enhancement using fft
    :param img: the original image
    :return: the enhanced image using fft
    """
    block_size = 32
    height, width = img.shape
    i = j = 0
    K = 0.1
    new = np.zeros(img.shape)
    while i < height:
        while j < width:
            tmp = np.zeros((block_size, block_size), dtype=img.dtype)
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    tmp[h-i][k-j] = img[h][k]
            # now tmp is a 32x32 (or less) image
            # fft enhancement
            tmp_frequency = np.fft.fft2(tmp)
            new_tmp = np.real(np.fft.ifft2(tmp_frequency*(np.abs(tmp_frequency)**K)))
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    new[h][k] = new_tmp[h-i][k-j]
            j += 32
        i += 32
        j = 0
    return new


def binarization(img: np.ndarray) -> np.ndarray:
    """
    Image binarization
    :param img: the original image
    :return: the binarized image
    """
    block_size = 16
    height, width = img.shape
    i = j = 0
    new = np.zeros(img.shape)
    while i < height:
        while j < width:
            tmp = np.zeros((block_size, block_size), dtype=img.dtype)
            count = 0
            pixel_sum = 0
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    tmp[h-i][k-j] = img[h][k]
                    count += 1
                    pixel_sum += img[h][k]
            mean = pixel_sum/count
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    if tmp[h-i][k-j] > mean:
                        new[h][k] = 1
                    else:
                        new[h][k] = 0
            j += 16
        i += 16
        j = 0
    return new


def ridge_thinning(img: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(img)
    skeleton = skeleton.astype(np.float)
    return skeleton


def block_direction_estimation(img: np.ndarray, block_size: int) -> np.ndarray:
    """
    Estimates the direction of each ridge and furrows using Hung least squares approximation
    and discard background blocks
    :param img: the original image
    :param block_size: the size of the block
    :return: the direction map
    """
    sobel_kernel_size = 5
    height, width = img.shape
    theta_map = []

    # partial derivatives
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

    for i in range(0, height+block_size, block_size):
        for j in range(0, width+block_size, block_size):
            block_coordinates, new_shape = neighbor_coordinates(seed_coordinates=(i, j),
                                                                kernel_size=block_size,
                                                                height=height, width=width)

            block_gx = np.array([gx[px[0], px[1]] for px in block_coordinates]).reshape(new_shape)
            block_gy = np.array([gy[px[0], px[1]] for px in block_coordinates]).reshape(new_shape)

            vx = np.sum(2*np.multiply(block_gx, block_gy))
            vy = np.sum(np.multiply(block_gx**2, block_gy**2))
            theta_map.append(0.5*np.arctan(np.divide(vy, vx+1e-6)))
    return theta_map


def gabor_filtering(img: np.ndarray, theta_map: np.ndarray, block_size: int) -> np.ndarray:
    """
    Gabor filtering
    :param img: the original image
    :param theta_map: the matrix containing ridge orientation
    :param block_size: the size of the Gabor filter
    :return:
    """
    height, width = img.shape


    return filtered_img


def roi_extraction(img: np.ndarray) -> np.ndarray:
    """
    Extracts the Region of interest from the original image.
    The ROI extraction process consists in subtracting the closed image from the opened image
    and then discarding the borders.
    :param img: the original image
    :return: the region of interest of the image
    """
    kernel_size = 3
    # Alternatives: cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE, cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
    # display_image(opening, title="Opening")
    # display_image(closing, title="Closing")

    roi = opening-closing
    # display_image(roi, title="ROI")

    # find ROI edges
    sum_columns = np.sum(roi, axis=0)
    sum_rows = np.sum(roi, axis=1)

    leftmost_index = np.nonzero(sum_columns)[0][0]
    rightmost_index = np.nonzero(sum_columns)[0][-1]
    highest_index = np.nonzero(sum_rows)[0][0]
    lowest_index = np.nonzero(sum_rows)[0][-1]

    # extract the interesting region
    roi_crop = roi[leftmost_index:rightmost_index,
               highest_index:lowest_index]
    # print("ROI shape: "+str(roi_crop.shape))
    return roi_crop


if __name__ == '__main__':
    # fingerprint = load_image(filename="SOCOFing/Real/29__F_Right_ring_finger.BMP",
    #                          cv2_read_param=0)
    fingerprint = load_image(filename="line.png",
                             cv2_read_param=0)
    display_image(img=fingerprint, cmap="gray", title="Original fingerprint")

    fingerprint = cv2.bitwise_not(fingerprint)
    # equalized = cv2.equalizeHist(fingerprint)
    # fft_enhanced = fft_enhancement(equalized)
    # binarized = binarization(fft_enhanced)
    # region_of_interest = roi_extraction(binarized)
    # thinned = ridge_thinning(binarized)

    direction_map = block_direction_estimation(img=fingerprint, block_size=16)
