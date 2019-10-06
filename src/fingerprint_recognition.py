#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Main module"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import cv2

from minutiae import find_lines, false_minutiae_removal, print_minutiae, crossing_numbers, inter_ridge_length, same_ridge, remove_minutiae, print_minutiae3
from utils import load_image
from utils import get_neighbor_coordinates
from utils import print_images
from utils import print_color_image
from typing import Tuple, Union
import matching

import gabor_filtering as gabor
import fingerprint_enhancement as enhancement


def pre_processing(img: np.ndarray):
    """
    Fingerprint image preprocessing operations
    :param img: the original image
    :return: the processed image
    """
    # image enhancement
    negated = cv2.bitwise_not(img)
    print_images([negated], ["original fingerprint"])
    denoised = cv2.fastNlMeansDenoising(negated, None, 15)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)
    normalized = enhancement.normalize(equalized)


    # gabor filtering
    ridge_orientation = gabor.get_orientation_map(normalized)
    ridge_frequency = gabor.get_frequency_map(normalized)
    a = np.mean(ridge_frequency)
    roi = enhancement.roi_mask(normalized)
    image = gabor.gabor_filter(normalized, ridge_orientation, ridge_frequency,
                               block_size=32, gabor_kernel_size=16)

    # extract ROI
    image = np.where(roi == 1.0, image, 1.0)
    binarized = enhancement.binarization(image)
    thinned = enhancement.ridge_thinning(binarized)
    print_images([thinned], ["thinned fingerprint"])
    return thinned, ridge_orientation, ridge_frequency


def get_freq(ridge_frequency1, ridge_frequency2):
    freq1 = 1 / np.mean(ridge_frequency1)
    freq2 = 1 / np.mean(ridge_frequency2)
    return freq1, freq2


def pre_processing2(img: np.ndarray):
    """
    no prints
    """
    # image enhancement
    negated = cv2.bitwise_not(img)
    denoised = cv2.fastNlMeansDenoising(negated, None, 15)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)
    normalized = enhancement.normalize(equalized)
    # gabor filtering
    ridge_orientation = gabor.get_orientation_map(normalized)
    ridge_frequency = gabor.get_frequency_map(normalized)
    a = np.mean(ridge_frequency)
    roi = enhancement.roi_mask(normalized)
    image = gabor.gabor_filter(normalized, ridge_orientation, ridge_frequency,
                               block_size=32, gabor_kernel_size=16)

    # extract ROI
    image = np.where(roi == 1.0, image, 1.0)
    binarized = enhancement.binarization(image)
    thinned = enhancement.ridge_thinning(binarized)
    return thinned, ridge_orientation, ridge_frequency


if __name__ == '__main__':
    fingerprint1 = load_image(filename="indice_dx_nico_1.jpg", cv2_read_param=0)
    fingerprint2 = load_image(filename="pollice_dx_luigi_2.jpg", cv2_read_param=0)
    processed_img1, ridge_orientation_map1, ridge_frequency1 = pre_processing(fingerprint1)
    processed_img2, ridge_orientation_map2, ridge_frequency2 = pre_processing(fingerprint2)

    minutiae1 = crossing_numbers(processed_img1, ridge_orientation_map1)
    minutiae2 = crossing_numbers(processed_img2, ridge_orientation_map2)
    minutiae11 = minutiae1.copy()
    minutiae22 = minutiae2.copy()
    ridge_identification_map1, labels1 = find_lines(processed_img1)
    ridge_identification_map2, labels2 = find_lines(processed_img2)
    freq1, freq2 = get_freq(ridge_frequency1, ridge_frequency2)
    minutiae1 = false_minutiae_removal(processed_img1, minutiae1, ridge_identification_map1, freq1/1.5)
    minutiae2 = false_minutiae_removal(processed_img2, minutiae2, ridge_identification_map2, freq2/1.5)
    minutiae1 = remove_minutiae(minutiae1)
    minutiae2 = remove_minutiae(minutiae2)
    minutiae11 = false_minutiae_removal(processed_img1, minutiae11, ridge_identification_map1, freq1)
    minutiae22 = false_minutiae_removal(processed_img2, minutiae22, ridge_identification_map2, freq2)
    minutiae11 = remove_minutiae(minutiae11)
    minutiae22 = remove_minutiae(minutiae22)
    print_minutiae(processed_img1, minutiae1, 255, 0, 0, "minutiae found")
    print_minutiae(processed_img1, minutiae11, 0, 255, 0, "false minutiae removed")
    print_minutiae(processed_img2, minutiae2, 255, 0, 0, "minutiae found")
    print_minutiae(processed_img2, minutiae22, 0, 255, 0, "false minutiae removed")
    msg = matching.match(processed_img1, processed_img2, minutiae1, minutiae2, minutiae11, minutiae22)
    print(msg)