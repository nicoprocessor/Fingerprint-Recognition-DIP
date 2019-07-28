#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Main module"""
from numpy.core.multiarray import ndarray

__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import logging
from skimage.morphology import skeletonize
from typing import Tuple, Union
import cv2

import utils
from utils import load_image
from utils import neighbor_coordinates
from utils import display_image
from utils import inclusive_range
from utils import print_images
from utils import print_color_image


# def fft_enhancement(img: np.ndarray) -> np.ndarray:
#     """
#     Image enhancement using fft
#     :param img: the original image
#     :return: the enhanced image using fft
#     """
#     block_size = 16
#     height, width = img.shape
#     i = j = 0
#     K = 0.45
#     new = np.zeros(img.shape)
#     while i < height:
#         while j < width:
#             tmp = np.zeros((block_size, block_size), dtype=img.dtype)
#             for h in range(i, min(i+block_size, height)):
#                 for k in range(j, min(j+block_size, width)):
#                     tmp[h-i][k-j] = img[h][k]
#             # now tmp is a 32x32 (or less) image
#             # fft enhancement
#             tmp_frequency = np.fft.fft2(tmp)
#             new_tmp = (np.real(np.fft.ifft2(tmp_frequency*(np.abs(tmp_frequency)**K))))/(32*32)
#             for h in range(i, min(i+block_size, height)):
#                 for k in range(j, min(j+block_size, width)):
#                     new[h][k] = new_tmp[h-i][k-j]
#             j += 32
#         i += 32
#         j = 0
#     return new


# def binarization(img: np.ndarray) -> np.ndarray:
#     """
#     Image binarization
#     :param img: the original image
#     :return: the binarized image
#     """
#     block_size = 32
#     height, width = img.shape
#     i = j = 0
#     new = np.zeros(img.shape)
#     while i < height:
#         while j < width:
#             tmp = np.zeros((block_size, block_size), dtype=img.dtype)
#             count = 0
#             pixel_sum = 0
#             for h in range(i, min(i+block_size, height)):
#                 for k in range(j, min(j+block_size, width)):
#                     tmp[h-i][k-j] = img[h][k]
#                     count += 1
#                     pixel_sum += img[h][k]
#             mean = pixel_sum/count
#             for h in range(i, min(i+block_size, height)):
#                 for k in range(j, min(j+block_size, width)):
#                     if tmp[h-i][k-j] > 1.2*mean and mean >= 10:
#                         new[h][k] = 1
#                     else:
#                         new[h][k] = 0
#             j += block_size
#         i += block_size
#         j = 0
#     return new


def otsu(img: np.ndarray) -> np.ndarray:
    """
    Otsu binarization
    :param img: the original image
    :return: the binarized image
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3/255


def ridge_thinning(img: np.ndarray) -> np.ndarray:
    """
    Ridge thinning or image skeletonization
    :param img: the original image
    :return: the skeleton of the image
    """
    skeleton = skeletonize(img)
    skeleton = skeleton.astype(np.float)
    return skeleton


def gabor_filtering(img: np.ndarray, block_size: int = 16,
                    gabor_kernel_size: Union[int, None] = None,
                    lpf_size: int = 5, sobel_kernel_size: int = 5,
                    precise_orientation_map: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates the direction of each ridge and furrows using Hung least squares approximation
    and discard background blocks
    :param img: the original image
    :param block_size: the size of the block. By default it is set at 16.
    :param gabor_kernel_size: the size of the Gabor kernel used to extract local features.
                By default it is set to be equal to the block size.
    :param lpf_size: the size of the low pass filter kernel used to improve the orientation ridge.
                By default it is set at 5.
    :param sobel_kernel_size: the size of the Sobel kernel used to extract partial derivatives.
                By default it is set at 5.
    :param precise_orientation_map: if True (default) uses a more precise algorithm for ridge direction estimation.
    :return: the filtered image
    """
    height, width = img.shape
    filtered_image = np.zeros(img.shape, dtype=img.dtype)
    orientation_map = {}
    phi_map = {}
    improved_orientation_map = {}

    if gabor_kernel_size is None:
        gabor_kernel_size = block_size

    # partial derivatives
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

    direction_map = np.zeros(shape=(int(np.ceil(height/block_size)), int(np.ceil(width/block_size))))

    for i in inclusive_range(block_size//2, height, block_size):
        for j in inclusive_range(block_size//2, width, block_size):
            neighborhood_coordinates, neighborhood_shape = neighbor_coordinates(seed_coordinates=(i, j),
                                                                                kernel_size=block_size,
                                                                                height=height, width=width)

            # block direction estimation
            image_block = np.array([img[px[0], px[1]] for px in neighborhood_coordinates]).reshape(neighborhood_shape)
            block_gx = np.array([gx[px[0], px[1]] for px in neighborhood_coordinates]).reshape(neighborhood_shape)
            block_gy = np.array([gy[px[0], px[1]] for px in neighborhood_coordinates]).reshape(neighborhood_shape)

            vx = np.sum(2*np.multiply(block_gx, block_gy))
            vy = np.sum(np.multiply(block_gx**2, block_gy**2))
            block_orientation = 0.5*np.arctan(np.divide(vy, vx+1e-6))  # scalar
            orientation_map[(i, j)] = block_orientation

            if precise_orientation_map:
                # smoothing & improving orientation estimation
                phi_map[i, j] = (np.cos(2*block_orientation), np.sin(2*block_orientation))  # (phi_x, phi_y)

                # convert to continuous vector field, by expanding each phi_block
                phi_x_block = phi_map[i, j][0]*np.ones(neighborhood_shape)
                phi_y_block = phi_map[i, j][1]*np.ones(neighborhood_shape)

                # low pass filtering with a median kernel, unit integral
                lpf_phi_x_block = cv2.blur(phi_x_block, (lpf_size, lpf_size))  # phi'_x
                lpf_phi_y_block = cv2.blur(phi_y_block, (lpf_size, lpf_size))  # phi'_y

                # save the current orientation to the ridge map (dictionary)
                improved_block_orientation = 0.5*np.arctan(np.divide(lpf_phi_y_block, lpf_phi_x_block))[0, 0]
                improved_orientation_map[(i, j)] = improved_block_orientation
            else:
                improved_block_orientation = block_orientation
                improved_orientation_map = orientation_map

            # Gabor filtering according to estimated ridge block orientation
            gabor_kernel = cv2.getGaborKernel((block_size, block_size), 1, improved_block_orientation, 10.0, 1,
                                              ktype=cv2.CV_32F)
            gabor_filtered_block = cv2.filter2D(image_block, -1, gabor_kernel). \
                reshape(image_block.shape[0]*image_block.shape[1], )

            # set the computed pixels in the destination image
            for index, px in enumerate(gabor_filtered_block):
                current_coordinates = neighborhood_coordinates[index]
                filtered_image[current_coordinates[0], current_coordinates[1]] = px

    # reconvert direction_map to np.array
    for coord, ridge_orientation in improved_orientation_map.items():
        direction_map[coord[0]//block_size, coord[1]//block_size] = ridge_orientation
    return filtered_image, direction_map


def clean(img):
    """
    Perform the clean morphological operation
    :param img: the original image
    :return: the cleaned image
    """
    height, width = img.shape
    new = np.zeros(img.shape, img.dtype)
    for i in range(height):
        for j in range(width):
            new[i][j] = img[i][j]
            if new[i][j] == 1:
                count = 0
                for h in range(max(0, i-2), min(height, i+3)):
                    for k in range(max(0, j-2), min(width, j+3)):
                        if img[h][k] == 1:
                            count += 1
                if count <= 2:
                    new[i][j] = 0
    return new


def skeleton_enhancement(img):
    """
    Perform skeleton enhancement performing a sequence of morphological operations
    :param img: the original skeletonized image
    :return: an improved version of the skeletonized image
    """
    tmp = clean(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel, iterations=1)
    enhanced_skeleton = ridge_thinning(closing)
    return enhanced_skeleton


def display_orientation_map(ridge_orientation: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Displays the vector field defined by the ridge orientation matrix
    :param ridge_orientation: the ridge orientation matrix
    :param block_size: the size of each block. By default it is set to 16
    :return:
    """
    # TODO lerp color
    block_offset = 0
    ridges_height, ridges_width = ridge_orientation.shape
    result_shape = (ridges_height*block_size, ridges_height*block_size)
    result_image = np.zeros(result_shape)

    # loop over every element of the ridge orientation map
    for i in range(ridges_height):
        for j in range(ridges_width):
            seed = (i*block_size+block_size//2, j*block_size+block_size//2)  # origin of the block
            current_orientation = ridge_orientation[i, j]
            slope = np.tan(current_orientation)
            intercept = (block_size//2)*(1-slope)

            # compute anchor point based on (0,0) block
            if abs(slope) > 1:  # the two anchor points will stay on the top and bottom of the window
                x_top = int(-intercept/(slope+1e-6))
                x_bottom = int((block_size+intercept)/(slope+1e-6))

                if slope > 0:
                    affine_anchor_points = [[x_top, 0, 1], [x_bottom, block_size, 1]]
                else:
                    affine_anchor_points = [[x_bottom, block_size, 1], [x_top, 0, 1]]
            elif slope == 0:
                affine_anchor_points = [[block_size//2, 0, 1], [block_size//2, block_size, 1]]
            else:  # the two anchor points will stay on the left and right of the window
                y_left = int(intercept)
                y_right = int(slope*block_size+intercept)

                if slope > 0:
                    affine_anchor_points = [[0, y_left, 1], [block_size, y_right, 1]]
                else:
                    affine_anchor_points = [[block_size, y_right, 1], [0, y_left, 1]]

            # translate anchor points according to current seed using affine transforms
            affine_translation_matrix = np.array([[1.0, 0.0, seed[0]-block_size//2],
                                                  [0.0, 1.0, seed[1]-block_size//2],
                                                  [0.0, 0.0, 1.0]])

            affine_anchor_start = np.array(affine_anchor_points[0])
            affine_anchor_end = np.array(affine_anchor_points[1])

            affine_anchor_start_translated = tuple(
                (np.dot(affine_translation_matrix, affine_anchor_start.transpose()).astype(np.int)))
            affine_anchor_end_translated = tuple(
                (np.dot(affine_translation_matrix, affine_anchor_end.transpose()).astype(np.int)))

            cv2.line(result_image, affine_anchor_start_translated[:-1],
                     affine_anchor_end_translated[:-1],
                     (255, 255, 255), 1)

    return result_image


def find_lines(img):
    """

    :param img:
    :return:
    """
    height, width = img.shape
    label_map = {}
    current_label = 1
    for i in range(height):
        for j in range(width):
            if img[i][j] == 1 and (i, j) not in label_map:
                label_map[(i, j)] = current_label
                label_line(img, i, j, label_map, current_label, height, width)
                current_label += 1
    return label_map, current_label


def label_line(img, i, j, label_map, label, height, width):
    """

    :param img:
    :param i:
    :param j:
    :param label_map:
    :param label:
    :param height:
    :param width:
    :return:
    """
    for h in range(max(0, i-1), min(height, i+2)):
        for k in range(max(0, j-1), min(width, j+2)):
            if img[h][k] == 1 and (h, k) not in label_map:
                label_map[(h, k)] = label
                # recursive call
                label_line(img, h, k, label_map, label, height, width)


def print_fingerprint_lines(img, label_map, labels):
    """
    Displays identified furrows using different colors
    :param img: the original image
    :param label_map: the label dictionary
    :param labels: the labels identified previously
    :return:
    """
    height, width = img.shape
    blank = np.zeros((height, width, 3), np.uint8)
    current = 0
    for l in range(1, labels+1):
        pos = [k for k, v in label_map.items() if v == l]
        if current != l:
            current = l
            r = np.random.randint(50, 255)
            g = np.random.randint(50, 255)
            b = np.random.randint(50, 255)
        for (i, j) in pos:
            blank[i][j] = (b, g, r)
    print_color_image(blank)


def count_pixels(img, i, j):
    height, width = img.shape
    count = 0
    for h in range(max(0, i - 1), min(height, i + 2)):
        for k in range(max(0, j - 1), min(width, j + 2)):
            if img[h][k] == 1:
                count = count+1
    return count


def find_ridges(img, label_map, labels):
    ridges = []
    for l in range(1, labels + 1):
        pos = [k for k, v in label_map.items() if v == l]
        for (i, j) in pos:
            if count_pixels(img, i, j) == 2:
                ridges.append((i, j))
    return ridges


def find_bifurcation(img, label_map, labels):
    bifurcation = []
    for l in range(1, labels + 1):
        pos = [k for k, v in label_map.items() if v == l]
        for (i, j) in pos:
            if count_pixels(img, i, j) == 4:
                bifurcation.append((i, j))
    return bifurcation


def print_minutia(img, ridges, b, g, r):
    height, width = img.shape
    blank = np.zeros((height, width, 3), np.uint8)
    for h in range(height):
        for k in range(width):
            if img[h][k] == 1:
                blank[h][k] = (255, 255, 255)
    for (i, j) in ridges:
        blank[i][j] = (b, g, r)
    print_color_image(blank)


#TODO IMPROVE PARAMETERS
def pre_processing(img):
    """
    Preprocessing operations on the image
    :param img: the original image
    :return: the processed image
    """
    negated = cv2.bitwise_not(img)
    denoised = cv2.fastNlMeansDenoising(negated, None, 10)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)
    #print_images([negated, denoised, equalized])
    gabor_filtered, ridge_map = gabor_filtering(img=equalized, block_size=16,
                                                precise_orientation_map=True)
    binarized = otsu(gabor_filtered)
    thinned = ridge_thinning(binarized)
    cleaned = clean(thinned)
    enhanced = skeleton_enhancement(cleaned)
    #print_images([gabor_filtered, binarized, cleaned])
    return enhanced


if __name__ == '__main__':
    fingerprint = load_image(filename="test4.jpg", cv2_read_param=0)

    processed_img = pre_processing(fingerprint)

    # label_map, labels = find_lines(processed_img)
    # ridges = find_ridges(processed_img, label_map, labels)
    # bifurcation = find_bifurcation(processed_img, label_map, labels)
    # print_minutia(processed_img, ridges, 0, 0, 255)
    # print_minutia(processed_img, bifurcation, 255, 0, 0)

