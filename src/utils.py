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
from pathlib import PurePath
import matplotlib.pyplot as plt


def load_image(filename: str, cv2_read_param: int = 0) -> np.ndarray:
    """
    Loads an image from the "res" folder
    :param filename: the name of the image file (format must be specified)
    :param cv2_read_param: the parameter for cv2.imread
        As a reference:
        cv2.IMREAD_COLOR (1): Loads a color image. Any transparency of image will be neglected
        cv2.IMREAD_GRAYSCALE (0): Loads image in grayscale mode
        cv2.IMREAD_UNCHANGED (-1): Loads image as such including alpha channel
    :return: the image read from the filesystem
    """
    filename = filename.split("/")
    filepath = PurePath(Path.cwd().parent, 'res')
    for f in filename:
        filepath = filepath.joinpath(f)

    # print("Filepath: "+str(filepath))
    assert (Path(filepath.as_posix()).exists() is True), "Wrong path: "+str(filepath)

    img = cv2.imread(str(filepath), cv2_read_param)
    assert (img is not None), "There was an issue loading the image"

    if cv2_read_param != 0:
        # invert channels from BGR to RGB
        img[:, :, :] = img[:, :, ::-1]
    return img


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


def display_image(img: np.ndarray, title: str = "Image", cmap: str = "gray"):
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


# TODO still an experiment
def print_images_args(**images):
    """
    Display several images in a single plot
    :param images: the images to print
    """
    for index, (title, image) in enumerate(images.items(), start=1):
        print(index, title)
        plt.subplot(1, len(images), index)
        plt.title(images[str(title)])
        plt.imshow(image, cmap='gray')
    plt.show()


def print_images(images):
    """
    Display several images in a single plot
    :param images: the images to print
    """
    for i in range(1, len(images)+1):
        plt.subplot(1, len(images), i)
        plt.imshow(images[i-1], cmap='gray')
    plt.show()
