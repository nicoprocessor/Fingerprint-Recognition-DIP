#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Utils functions for image manipulation"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import cv2
import numpy as np
from pathlib import Path
from pathlib import PurePath
from typing import Tuple, List, Union
import matplotlib.pyplot as plt

Pair = Tuple[int, int]
Scalar = Union[int, float, np.float32]
Color = Tuple[np.uint8, np.uint8, np.uint8]


def lerp_color(start: Color = (0, 0, 0), end: Color = (255, 255, 255),
               percentage: float = 0.5) -> Color:
    """
    Linear color interpolation
    :param start: the starting color of the gradient. By default set to black.
    :param end: the final color of the gradient. By default set to white.
    :param percentage: the gradient percentage, expressed as a value between 0.0 and 1.0. By default it is set to 0.5.
    :return: the interpolated color
    """
    lerp_r = start[0]+(end[0]-start[0])*percentage
    lerp_g = start[1]+(end[1]-start[1])*percentage
    lerp_b = start[2]+(end[2]-start[2])*percentage
    return lerp_r, lerp_g, lerp_b


def inclusive_range(start: Scalar, stop: Scalar, step: Scalar = 1):
    """
    Like a normal range, but with inclusive end
    :param start: starting element
    :param stop: final element
    :param step: range step
    """
    assert ((isinstance(start, int) or isinstance(start, float)) and
            (isinstance(stop, int) or isinstance(stop, float)) and
            (isinstance(step, int) or isinstance(step, float))), "Wrong argument type"
    current = start
    while current <= stop:
        yield current
        current += step


def check_membership_in_sorted_set_and_pop(element: Scalar, sorted_list: List[Scalar]) -> Tuple[bool, List[Scalar]]:
    """
    Check if an element is in the sorted list without looping on the entire list.
    If the element is in the list, then returns the list without the elements
    that are smaller than the searched element.
    :param element: the searched element
    :param sorted_list: the sorted list
    :return: True if the element is in the list, False otherwise.
    Moreover returns the list without the elements that are smaller than the searched element.
    """
    assert (len(sorted_list) > 0), "Empty list"
    assert (isinstance(element, int) or isinstance(element, float)), "Can only search for single elements"
    assert (sorted(sorted_list) == sorted_list), "List is unsorted"

    while True:
        current_element = sorted_list[0]
        if element == current_element:
            return True, sorted_list
        elif current_element > element:
            return False, sorted_list
        else:
            sorted_list = sorted_list[1:]  # pop head


def neighbor_coordinates(seed_coordinates: Pair, kernel_size: int, height: int, width: int,
                         allow_diagonals: bool = True, include_seed=True) -> Tuple[List[Pair], Pair]:
    """
    Compute the list of the coordinates of the neighbors, given the coordinates of the center, in a 2D matrix
    :param seed_coordinates: the coordinates of the central element
    :param kernel_size: the size of the squared kernel
    :param height: the number of rows
    :param width: the number of columns
    :param allow_diagonals: if True computes the neighbors in all 8 directions, if False computes only N-S-E-W
    :param include_seed: decide whether the seed (central element) must be returned
    :return: the list of coordinates around the given element and the size of the neighborhood
    """
    pad = kernel_size//2

    # compute border pixels
    leftmost = max(0, seed_coordinates[1]-pad)
    rightmost = min(width-1, seed_coordinates[1]+pad+1)
    highest = max(0, seed_coordinates[0]-pad)
    lowest = min(height-1, seed_coordinates[0]+pad+1)

    shape = (lowest-highest, rightmost-leftmost)

    if allow_diagonals:
        neighbors = [(i, j) for j in range(leftmost, rightmost)
                     for i in range(highest, lowest)]
    else:
        pivot_row = set([(i, seed_coordinates[1]) for i in range(highest, lowest)])
        pivot_col = set([(seed_coordinates[0], j) for j in range(leftmost, rightmost)])
        neighbors = list(pivot_row.union(pivot_col))
    if not include_seed:
        neighbors.remove(seed_coordinates)
    return neighbors, shape


def load_image(filename: str, cv2_read_param: int = 0) -> np.ndarray:
    """
    Loads an image from the "res" folder of the project
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


def print_color_image(image):
    image = image[:, :, ::-1]
    plt.imshow(image, cmap='gray')
    plt.show()
