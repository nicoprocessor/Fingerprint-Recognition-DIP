#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Main module"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import cv2

from utils import load_image
from utils import display_image
from utils import print_images


def extract_roi(img: np.ndarray) -> np.ndarray:
    """
    Extracts the Region Of Interest from the original image.
    The ROI extraction process consists in subtracting the closed image from the opened image
    and then discarding the borders
    :param img: the original image
    :return: the region of interest of the image
    """
    # Alternatives: cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE
    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    roi = opening-closing

    # TODO borders removal

    return img


def fft_enhancement(img: np.ndarray) -> np.ndarray:
    """
    image enhancement using fft
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
            tmp  = np.zeros((block_size, block_size), dtype=img.dtype)
            for h in range(i, min(i + block_size, height)):
                for k in range(j, min(j + block_size, width)):
                    tmp[h-i][k-j] = img[h][k]
            ##now tmp is a 32x32 (or less) image
            ##fft enhancement
            tmp_frequency = np.fft.fft2(tmp)
            new_tmp = np.real(np.fft.ifft2(tmp_frequency*(np.abs(tmp_frequency)**K)))
            for h in range(i, min(i + block_size, height)):
                for k in range(j, min(j + block_size, width)):
                    new[h][k] = new_tmp[h - i][k - j]
            j+=32
        i+=32
        j=0
    return new


def binarization(img: np.ndarray) -> np.ndarray:
    """
    image binarization
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
            for h in range(i, min(i + block_size, height)):
                for k in range(j, min(j + block_size, width)):
                    tmp[h - i][k - j] = img[h][k]
                    count += 1
                    pixel_sum += img[h][k]
            mean = pixel_sum/count
            for h in range(i, min(i + block_size, height)):
                for k in range(j, min(j + block_size, width)):
                    if tmp[h - i][k - j] > mean:
                        new[h][k] = 1
                    else: new[h][k] = 0
            j += 16
        i += 16
        j = 0
    return new

if __name__ == '__main__':
    fingerprint = load_image(filename="path/29__F_Right_ring_finger.BMP", cv2_read_param=0)
    fingerprint = cv2.bitwise_not(fingerprint)
    equalized = cv2.equalizeHist(fingerprint)
    fft_enhanced = fft_enhancement(equalized)
    print_images([fingerprint, equalized, fft_enhanced, binarization(fft_enhanced)])








