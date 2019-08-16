#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Main module"""
from numpy.core.multiarray import ndarray

__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
from typing import Tuple, Union
import cv2

from utils import load_image
from utils import neighbor_coordinates
from utils import inclusive_range
from utils import print_images
from utils import print_color_image
import matplotlib.pyplot as plt

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
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(32, 32))
    equalized = clahe.apply(denoised)
    #normalized = enhancement.normalize2(equalized, 20, 5)
    normalized = enhancement.normalize(equalized)

    # gabor filtering
    ridge_orientation = gabor.get_orientation_map(normalized)
    ridge_frequency = gabor.get_frequency_map(normalized)
    roi = enhancement.roi_mask(normalized)
    image = gabor.gabor_filter(normalized, ridge_orientation, ridge_frequency,
                               block_size=32, gabor_kernel_size=16)

    # extract ROI
    image = np.where(roi == 1.0, image, 1.0)
    print_images([denoised, equalized, image])
    print_images([enhancement.clean(enhancement.ridge_thinning(enhancement.binarization(image)))])



if __name__ == '__main__':
    fingerprint = load_image(filename="test4.jpg", cv2_read_param=0)

    pre_processing(fingerprint)

    # label_map, labels = find_lines(processed_img)
    # ridges = find_ridges(processed_img, label_map, labels)
    # bifurcation = find_bifurcation(processed_img, label_map, labels)
    # print_minutia(processed_img, ridges, 0, 0, 255)
    # print_minutia(processed_img, bifurcation, 255, 0, 0)
