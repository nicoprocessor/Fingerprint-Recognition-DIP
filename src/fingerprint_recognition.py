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


if __name__ == '__main__':
    fingerprint = load_image(filename="SOCOFing/Real/1__M_Left_index_finger.BMP", cv2_read_param=0)
    display_image(img=fingerprint, cmap="gray", title="Original fingerprint")
