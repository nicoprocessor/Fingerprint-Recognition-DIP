#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Utility functions for fingerprint minutiae analysis"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import scipy.sparse as sparse

from utils import get_neighbor_coordinates, inverse_dictionary, print_color_image, print_images
from utils import Coordinate

from typing import Tuple, Dict, List


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
    for h in range(max(0, i-1), min(height, i+2)):
        for k in range(max(0, j-1), min(width, j+2)):
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

    for l in range(1, labels+1):
        pos = [k for k, v in label_map.items() if v == l]  # extract pixel coordinates that share the same label
        if current != l:
            current = l
            r = np.random.randint(50, 255)
            g = np.random.randint(50, 255)
            b = np.random.randint(50, 255)
        # assign to each pixel in the same ridge the selected random color
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
    height, width = image.shape
    count = 0

    # my implementation using Numpy
    # neighborhood_coordinates, _ = neighbor_coordinates(seed_coordinates=(i, j), kernel_size=block_size,
    #                                                   height=height, width=width)
    # return np.sum(np.array([gy[px[0], px[1]] for px in neighborhood_coordinates]))

    for h in range(max(0, i-1), min(height, i+2)):
        for k in range(max(0, j-1), min(width, j+2)):
            if image[h][k] == 1:
                count = count+1
    return count


def find_terminations(skeleton: np.ndarray, label_map, labels: int) -> Coordinate:
    """
    Find ridge termination in the fingerprint image
    :param skeleton: the fingerprint skeleton
    :param label_map:
    :param labels:
    :return:
    """
    terminations = []
    for l in range(1, labels+1):
        pos = [k for k, v in label_map.items() if v == l]
        for (i, j) in pos:
            if count_pixels(skeleton, i, j) == 2:
                terminations.append((i, j))
    return terminations


def find_bifurcations(skeleton: np.ndarray, label_map, labels: int) -> Coordinate:
    """
    Find bifurcations in the fingerprint image
    :param skeleton: the fingerprint skeleton
    :param label_map:
    :param labels:
    :return:
    """
    bifurcations = []
    for l in range(1, labels+1):
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
    return np.mean(inter_ridge_distance)


def false_minutiae_removal(skeleton: np.ndarray,
                           ridge_map: Dict[Coordinate, int],
                           bifurcations: np.ndarray,
                           terminations: np.ndarray,
                           orientation_map: np.ndarray,
                           orientation_angle_threshold: float = np.pi/3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove false minutiae from the fingerprint skeleton
    :param skeleton: the fingerprint skeleton
    :param ridge_map: the ridge map that binds each pixel in a ridge with the corresponding ridge identifier
    :param bifurcations: the coordinates of every ridge bifurcation in the skeleton
    :param terminations: the coordinates of every ridge termination in the skeleton
    :param orientation_map: the matrix containing the direction of each ridge
    :return: the bifurcations and terminations coordinates, without false records
    """
    height, width = skeleton.shape
    false_minutiae_threshold = inter_ridge_length(skeleton)
    print("False minutiae threshold: "+str(false_minutiae_threshold))  # very low value! Find a better solution

    false_bifurcations, false_terminations = [], []
    # real_bifurcations, real_terminations = [], []

    # convert minutiae matrices from sparse to full format
    bifurcations_rows = [x[0] for x in bifurcations]
    bifurcations_cols = [x[1] for x in bifurcations]
    bifurcation_image = sparse.coo_matrix((np.ones(len(bifurcations)), (bifurcations_rows, bifurcations_cols)),
                                          shape=skeleton.shape).toarray()

    terminations_rows = [x[0] for x in terminations]
    terminations_cols = [x[1] for x in terminations]
    terminations_image = sparse.coo_matrix((np.ones(len(terminations)), (terminations_rows, terminations_cols)),
                                           shape=skeleton.shape).toarray()

    # leave for future use
    # inverse_ridge_map = inverse_dictionary(original_dict=ridge_map, unique_values=False)  # Dict[int, List[Coordinate]]

    false_bifurcation_removal(bifurcations=bifurcations, bifurcation_image=bifurcation_image, ridge_map=ridge_map)
    return false_bifurcations, false_terminations


def false_bifurcation_removal(height: int, weight: int,
                              bifurcations: List[Coordinate],
                              bifurcation_image: np.ndarray,
                              ridge_map: Dict[Coordinate, int]):
    """
    Remove false bifurcations in the fingerprint skeleton
    :param height: skeleton height
    :param weight: skeleton weight
    :param bifurcations: the list of bifurcation coordinates
    :param bifurcation_image: the image containing "1" where a bifurcation is located
    :param ridge_map: a dictionary mapping each pixel of a ridge to its unique identifier
    :return: the bifurcations coordinates, the bifurcation image and the list of false bifurcations
    """
    # current_bifurcation: Coordinate
    for current_bifurcation in bifurcations:
        # find other minutiae in the neighborhood -> no need to check distance
        neighbor_coordinates, _ = get_neighbor_coordinates(seed_coordinates=current_bifurcation,
                                                           kernel_size=np.floor(false_minutiae_threshold),
                                                           width=width, height=height, include_seed=False)
        # neighbor: Coordinate
        for neighbor in neighbor_coordinates:
            if bifurcation_image(neighbor[0], neighbor[1]) == 1:  # found another bifurcation in the neighborhood
                if ridge_map[neighbor] == ridge_map[current_bifurcation]:  # both minutiae belong to the same ridge
                    # remove both of them from the original list
                    bifurcations.remove(current_bifurcation)
                    bifurcations.remove(neighbor)

                    # update matrices
                    bifurcation_image[current_bifurcation[0], current_bifurcation[1]] = 0
                    bifurcation_image[neighbor[0], neighbor[1]] = 0

                    false_bifurcations.append(current_bifurcation)
                    false_bifurcations.append(neighbor)

            elif terminations_image(neighbor[0], neighbor[1]) == 1:  # found a termination in the neighborhood
                if ridge_map[neighbor] == ridge_map[current_bifurcation]:  # both minutiae belong to the same ridge
                    # remove both of them from the original list
                    bifurcations.remove(current_bifurcation)
                    terminations.remove(neighbor)

                    # update matrices
                    bifurcation_image[current_bifurcation[0], current_bifurcation[1]] = 0
                    terminations_image[neighbor[0], neighbor[1]] = 0

                    false_bifurcations.append(current_bifurcation)
                    false_terminations.append(neighbor)
    return bifucations, false_bifurcations, bifurcation_image


def false_terminations_removal():
    pass
