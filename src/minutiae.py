#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Utility functions for fingerprint minutiae analysis"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np

from utils import print_images
from typing import Tuple, Dict
from utils import print_color_image
from utils import neighbor_coordinates
from utils import Vector


def find_lines(skeleton: np.ndarray):
    """

    :param skeleton: the fingerprint skeleton
    :return:
    """
    height, width = skeleton.shape
    label_map = {}
    current_label = 1
    for i in range(height):
        for j in range(width):
            if skeleton[i][j] == 1 and (i, j) not in label_map:
                label_map[(i, j)] = current_label
                label_line(skeleton, i, j, label_map, current_label, height, width)
                current_label += 1
    return label_map, current_label


def label_line(skeleton: np.ndarray, i: int, j: int, label_map, label, height: int, width: int):
    """
    Recursively labels each ridge in the fingerprint skeleton
    :param skeleton: the fingerprint skeleton
    :param i: the current row
    :param j: the current column
    :param label_map:
    :param label: the labels found until this point
    :param height: the height of the image
    :param width: the width of the image
    """
    for h in range(max(0, i - 1), min(height, i + 2)):
        for k in range(max(0, j - 1), min(width, j + 2)):
            if skeleton[h][k] == 1 and (h, k) not in label_map:
                label_map[(h, k)] = label
                # recursive call
                label_line(skeleton, h, k, label_map, label, height, width)


def print_fingerprint_lines(image: np.ndarray, label_map, labels: int):
    """
    Displays identified furrows using different colors
    :param image: the original image
    :param label_map: the label dictionary
    :param labels: the labels identified previously
    """
    height, width = image.shape
    blank = np.zeros((height, width, 3), np.uint8)
    current, b, g, r = 0, 0, 0, 0

    for l in range(1, labels + 1):
        pos = [k for k, v in label_map.items() if v == l]
        if current != l:
            current = l
            r = np.random.randint(50, 255)
            g = np.random.randint(50, 255)
            b = np.random.randint(50, 255)
        for (i, j) in pos:
            blank[i][j] = (b, g, r)
    print_color_image(blank)


def count_pixels(image: np.ndarray, i: int, j: int, block_size: int = 3) -> int:
    """
    Count all the non-zero pixels in the 3-by-3 window centered at the specified location
    :param image: the original image
    :param i: the row of the seed point
    :param j: the column of the seed point
    :param block_size: the size of the window
    :return: the sum of non-zero pixels in the neighborhood
    """
    # TODO numpy?
    height, width = image.shape
    count = 0

    # my implementation using Numpy
    # neighborhood_coordinates, _ = neighbor_coordinates(seed_coordinates=(i, j), kernel_size=block_size,
    #                                                   height=height, width=width)
    # return np.sum(np.array([gy[px[0], px[1]] for px in neighborhood_coordinates]))

    for h in range(max(0, i - 1), min(height, i + 2)):
        for k in range(max(0, j - 1), min(width, j + 2)):
            if image[h][k] == 1:
                count = count + 1
    return count


def find_terminations(skeleton: np.ndarray, label_map, labels: int) -> Vector:
    """
    Find ridge termination in the fingerprint image
    :param skeleton: the fingerprint skeleton
    :param label_map:
    :param labels:
    :return:
    """
    terminations = []
    for l in range(1, labels + 1):
        pos = [k for k, v in label_map.items() if v == l]
        for (i, j) in pos:
            if count_pixels(skeleton, i, j) == 2:
                terminations.append((i, j))
    return terminations


def find_bifurcations(skeleton: np.ndarray, label_map, labels: int) -> Vector:
    """
    Find bifurcations in the fingerprint image
    :param skeleton: the fingerprint skeleton
    :param label_map:
    :param labels:
    :return:
    """
    bifurcations = []
    for l in range(1, labels + 1):
        pos = [k for k, v in label_map.items() if v == l]
        for (i, j) in pos:
            if count_pixels(skeleton, i, j) == 4:
                bifurcations.append((i, j))
    return bifurcations


def print_minutiae(skeleton: np.ndarray, ridges, b: int, g: int, r: int):
    """
    Highlight minutiae points in the fingerprint skeleton
    :param skeleton: the fingerprint skeleton
    :param ridges: the ridge map
    :param b: the blue color channel intensity value
    :param g: the green color channel intensity value
    :param r: the red color channel intensity value
    """
    height, width = skeleton.shape
    blank = np.zeros((height, width, 3), np.uint8)

    for h in range(height):
        for k in range(width):
            if skeleton[h][k] == 1:
                blank[h][k] = (255, 255, 255)
    for (i, j) in ridges:
        blank[i][j] = (b, g, r)
    print_color_image(blank)


def inter_ridge_length(skeleton: np.ndarray) -> float:
    inter_ridge_distance = np.mean(skeleton, axis=1)  # average of all the pixels along rows
    false_minutiae_threshold = np.mean(inter_ridge_distance)


def false_minutiae_removal(skeleton: np.ndarray,
                           bifurcation_map: np.ndarray,
                           termination_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove false minutiae from the fingerprint skeleton
    :param skeleton: the fingerprint skeleton
    :param bifurcation_map: the coordinates of each ridge bifurcation
    :param termination_map: the coordinates of every ridge termination
    :return: the bifurcations and terminations coordinates, without false recordings
    """
    height, width = skeleton.shape
    false_minutiae_threshold = inter_ridge_length(skeleton)
    bifurcation_image = np.zeros(shape=skeleton.shape, dtype=int)
    termination_image = np.zeros(shape=skeleton.shape, dtype=int)

    for b in bifurcation_map:
        bifurcation_image[b[0], b[1]] = 1

    for t in termination_map:
        termination_image[t[0], t[1]] = 1

    for h in range(height):
        for k in range(width):
            if bifurcation_image[h,k] == 1:
                 pass
    pass
