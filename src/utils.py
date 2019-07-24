#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Utils functions for image manipulation"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt


def load_image(filename: str, cv2_read_param: int = 0) -> Tuple[str, np.ndarray]:
    """
    Loads an image from the data folder using cv2.imread
    :param filename: the name of the image file (format must be specified)
    :param cv2_read_param: the parameter for cv2.imread
        As a reference:
        cv2.IMREAD_COLOR (1): Loads a color image. Any transparency of image will be neglected
        cv2.IMREAD_GRAYSCALE (0): Loads image in grayscale mode
        cv2.IMREAD_UNCHANGED (-1): Loads image as such including alpha channel
    :return: the image read from the file and its name
    """
    filepath = Path.cwd()/'data'/filename
    print("Filepath:"+str(filepath))
    # assert (filepath.exists() is True), "Wrong path: "+str(filepath)

    img_name = str(os.path.splitext(filename)[0])
    # img_basename = str(Path.cwd() / os.path.splitext(filename)[0])
    img = cv2.imread(str(filepath), cv2_read_param)
    assert (img is not None), "There was an issue loading the image"

    if cv2_read_param != 0:
        # invert channels from BGR to RGB
        img[:, :, :] = img[:, :, ::-1]
    return img_name, img


def save_image(filename: str, img: np.ndarray):
    """
    Save an image in the current directory
    :param filename: the name of the destination file
    :param img: the image that has to be saved
    """
    filepath = Path.cwd()/'data'/filename
    if len(img.shape) > 2:
        img[:, :, :] = img[:, :, ::-1]  # invert channels
    # assert (filepath.exists() is True), "Wrong path"
    cv2.imwrite(str(filepath), img)


def display_image(img: np.ndarray, cmap: str, title: str):
    """
    Display an image using matplotlib.imshow()
    :param img: original image
    :param cmap: the colormap passed to the matplotlib.imshow() function
    :param title: the title of the plot
    """
    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()