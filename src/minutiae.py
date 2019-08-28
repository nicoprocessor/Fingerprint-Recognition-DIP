#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Utility functions for fingerprint minutiae analysis"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import scipy.sparse as sparse

from utils import get_neighbor_coordinates, print_color_image, print_images
from utils import Coordinate

from typing import Tuple, Dict, List


def crossing_numbers(skeleton: np.ndarray, orientation_map):
    """
    :param skeleton: the fingerprint skeleton
    :return:
    """
    height, width = skeleton.shape
    minutiae = []
    for i in range(height-1):
        for j in range(width-1):
            if skeleton[i][j] == 1.0:
                if j == 130 and i == 392:
                    print('ok')
                CN = 0
                P = [skeleton[i][j+1], skeleton[i-1][j+1], skeleton[i-1][j], skeleton[i-1][j-1], skeleton[i][j-1], skeleton[i+1][j-1], skeleton[i+1][j], skeleton[i+1][j+1]]
                for k in range(len(P)):
                    CN += np.abs(P[k]-P[(k+1)%8])
                CN = CN * 0.5
                if CN == 1:
                    minutiae.append((i, j, CN, orientation_map[i][j]))
                if CN == 3:
                    minutiae.append((i, j, CN, orientation_map[i][j]))
    return minutiae


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


# def find_terminations(skeleton: np.ndarray, label_map, orientation_map, labels: int) -> Coordinate:
#     """
#     Find ridge termination in the fingerprint image
#     :param skeleton: the fingerprint skeleton
#     :param label_map:
#     :param labels:
#     :return:
#     """
#     terminations = []
#     for l in range(1, labels+1):
#         pos = [k for k, v in label_map.items() if v == l]
#         for (i, j) in pos:
#             if count_pixels(skeleton, i, j) == 2:
#                 terminations.append((i, j, orientation_map[i][j]))
#     return terminations
#
#
# def find_bifurcations(skeleton: np.ndarray, label_map, orientation_map, labels: int) -> Coordinate:
#     """
#     Find bifurcations in the fingerprint image
#     :param skeleton: the fingerprint skeleton
#     :param label_map:
#     :param labels:
#     :return:
#     """
#     bifurcations = []
#     for l in range(1, labels+1):
#         pos = [k for k, v in label_map.items() if v == l]
#         for (i, j) in pos:
#             if count_pixels(skeleton, i, j) == 4:
#                 bifurcations.append((i, j, orientation_map[i][j]))
#     return bifurcations


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
    for (i, j, _, _) in ridges:
        # for h in range(max(0, i - 1), min(height, i + 2)):
        #     for k in range(max(0, j - 1), min(width, j + 2)):
                blank[i][j] = (b, g, r)
    print_color_image(blank)


#TODO
def inter_ridge_length(skeleton: np.ndarray, roi) -> float:
    height, width = skeleton.shape
    d = []
    for i in range(height):
        count = 0
        for j in range(width):
            if roi[i][j] == 1.0:
                count += 1
        if count != 0:
            d.append(np.sum(skeleton[i]) / count)
    D = np.mean(d)
    return D


def same_ridge(minutia1, minutia2, ridge_identification_map):
    x1, y1, _, _ = minutia1
    x2, y2, _, _ = minutia2
    v1 = ridge_identification_map.get((x1, y1))
    v2 = ridge_identification_map.get((x2, y2))
    return v1 == v2


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

    false_bifurcations_removal(height=height, width=width, bifurcations=bifurcations, terminations=terminations,
                               bifurcations_image=bifurcation_image, terminations_image=terminations_image,
                               threshold=false_minutiae_threshold, ridge_map=ridge_map)
    false_terminations_removal(height=height, width=width, )
    return false_bifurcations, false_terminations


def false_bifurcations_removal(height: int, width: int,
                               bifurcations: List[Coordinate],
                               terminations: List[Coordinate],
                               bifurcations_image: np.ndarray,
                               terminations_image: np.ndarray,
                               threshold: float,
                               ridge_map: Dict[Coordinate, int]) -> Tuple[List[Coordinate],
                                                                          List[Coordinate], List[Coordinate], List[
                                                                              Coordinate], np.ndarray, np.ndarray]:
    """
    Remove false bifurcations in the fingerprint skeleton
    :param height: skeleton height
    :param width: skeleton weight
    :param bifurcations: the list of bifurcation coordinates
    :param terminations: the list of termination coordinates
    :param bifurcations_image: an image containing "1" where a bifurcation is located
    :param terminations_image: an image containing "1" where a termination is located
    :param ridge_map: a dictionary mapping each pixel of a ridge to its unique identifier
    :param threshold: the size of the neighborhood around a termination
    :return: the bifurcations and terminations coordinates, the bifurcation and terminations image and
    the list of false bifurcations and terminations
    """
    false_bifurcations, false_terminations = [], []

    # current_bifurcation: Coordinate
    for current_bifurcation in bifurcations:
        # find other minutiae in the neighborhood -> no need to check distance
        neighbor_coordinates, _ = get_neighbor_coordinates(seed_coordinates=current_bifurcation,
                                                           kernel_size=np.floor(threshold),
                                                           width=width, height=height, include_seed=False)
        # neighbor: Coordinate
        for neighbor in neighbor_coordinates:
            if bifurcations_image(neighbor[0], neighbor[1]) == 1:  # found another bifurcation in the neighborhood
                if ridge_map[neighbor] == ridge_map[current_bifurcation]:  # both minutiae belong to the same ridge
                    # remove both of them from the original list
                    bifurcations.remove(current_bifurcation)
                    bifurcations.remove(neighbor)

                    # update matrices
                    bifurcations_image[current_bifurcation[0], current_bifurcation[1]] = 0
                    bifurcations_image[neighbor[0], neighbor[1]] = 0

                    false_bifurcations.append(current_bifurcation)
                    false_bifurcations.append(neighbor)

            elif terminations_image(neighbor[0], neighbor[1]) == 1:  # found a termination in the neighborhood
                if ridge_map[neighbor] == ridge_map[current_bifurcation]:  # both minutiae belong to the same ridge
                    # remove both of them from the original list
                    bifurcations.remove(current_bifurcation)
                    terminations.remove(neighbor)

                    # update matrices
                    bifurcations_image[current_bifurcation[0], current_bifurcation[1]] = 0
                    terminations_image[neighbor[0], neighbor[1]] = 0

                    false_bifurcations.append(current_bifurcation)
                    false_terminations.append(neighbor)
    return bifurcations, terminations, false_bifurcations, false_teminations, bifurcations_image, terminations_image


# TODO test
def false_terminations_removal(height: int, width: int,
                               terminations: List[Coordinate],
                               terminations_image: np.ndarray,
                               ridge_map: Dict[Coordinate, int],
                               threshold: float,
                               orientation_map: np.ndarray,
                               ridge_orientation_threshold: float = np.pi/6):
    """
    Remove false bifurcations in the fingerprint skeleton
    :param height: skeleton height
    :param width: skeleton weight
    :param terminations: the list of termination coordinates
    :param terminations_image: an image containing "1" where a termination is located
    :param ridge_map: a dictionary mapping each pixel of a ridge to its unique identifier
    :param threshold: the size of the neighborhood around a termination
    :param orientation_map: a matrix containing the directions of the ridges
    :return: the terminations coordinates, the terminations image and the list of false terminations
    """
    false_terminations = []

    # current_bifurcation: Coordinate
    for current_termination in terminations:
        # find other minutiae in the neighborhood -> no need to check distance
        neighbor_coordinates, _ = get_neighbor_coordinates(seed_coordinates=current_termination,
                                                           kernel_size=np.floor(threshold),
                                                           width=width, height=height, include_seed=False)
        # neighbor: Coordinate
        # TODO change to the according criteria
        for neighbor in neighbor_coordinates:
            if bifurcations_image(neighbor[0], neighbor[1]) == 1:  # found another bifurcation in the neighborhood
                if ridge_map[neighbor] == ridge_map[current_termination]:  # both minutiae belong to the same ridge
                    # remove both of them from the original list
                    bifurcations.remove(current_termination)
                    bifurcations.remove(neighbor)

                    # update matrices
                    bifurcations_image[current_termination[0], current_termination[1]] = 0
                    bifurcations_image[neighbor[0], neighbor[1]] = 0

                    false_bifurcations.append(current_termination)
                    false_bifurcations.append(neighbor)

            elif terminations_image(neighbor[0], neighbor[1]) == 1:  # found a termination in the neighborhood
                if ridge_map[neighbor] == ridge_map[current_termination]:  # both minutiae belong to the same ridge
                    # remove both of them from the original list
                    bifurcations.remove(current_termination)
                    terminations.remove(neighbor)

                    # update matrices
                    bifurcations_image[current_termination[0], current_termination[1]] = 0
                    terminations_image[neighbor[0], neighbor[1]] = 0

                    false_bifurcations.append(current_termination)
                    false_terminations.append(neighbor)
    return terminations, false_teminations, terminations_image
