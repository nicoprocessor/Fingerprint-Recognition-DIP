#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Main module"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import cv2

from minutiae import find_lines, false_minutiae_removal, print_minutiae, crossing_numbers, inter_ridge_length, same_ridge
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
    denoised = cv2.fastNlMeansDenoising(negated, None, 15)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)
    normalized = enhancement.normalize(equalized)
    #print_images([negated, denoised, normalized])

    # gabor filtering
    ridge_orientation = gabor.get_orientation_map(normalized)
    ridge_frequency = gabor.get_frequency_map(normalized)
    roi = enhancement.roi_mask(normalized)
    image = gabor.gabor_filter(normalized, ridge_orientation, ridge_frequency,
                               block_size=32, gabor_kernel_size=16)

    # extract ROI
    image = np.where(roi == 1.0, image, 1.0)
    binarized = enhancement.binarization(image)
    thinned = enhancement.ridge_thinning(binarized)
    # print_images([image, enhancement.ridge_thinning(enhancement.binarization(image))])
    return thinned, ridge_orientation


if __name__ == '__main__':
    fingerprint1 = load_image(filename="thumb1.jpg", cv2_read_param=0)
    processed_img1, ridge_orientation_map1 = pre_processing(fingerprint1)
    minutiae = crossing_numbers(processed_img1, ridge_orientation_map1)
    print_minutiae(processed_img1, minutiae, 255, 0, 0)
    ridge_identification_map1, labels1 = find_lines(processed_img1)
    print('ok')
    a = same_ridge(minutiae[0], minutiae[2], ridge_identification_map1)
    print(a)