#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Main module"""

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
#                     if tmp[h-i][k-j] > 1.1*mean and mean >= 5:
#                         new[h][k] = 1
#                     else:
#                         new[h][k] = 0
#             j += block_size
#         i += block_size
#         j = 0
#     return new


def otsu(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3/255


def ridge_thinning(img: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(img)
    skeleton = skeleton.astype(np.float)
    return skeleton


# TODO display block orientation map
def display_orientation_map(ridge_orientation: np.ndarray, block_size: int) -> np.ndarray:
    """
    Displays the vector field defined by the ridge orientation matrix
    :param ridge_orientation: the ridge orientation matrix
    :param block_size: the size of each block
    :return:
    """
    result_shape = (ridge_orientation.shape[0]*block_size, ridge_orientation.shape[0]*block_size)
    result_image = np.zeros(result_shape)

    # loop over every element of the ridge orientation map

    # convert the current angle to a line slope

    # translate the line according to the block position
    pass


def gabor_filtering_block_level(img: np.ndarray, block_size: int = 16,
                                gabor_kernel_size: Union[int, None] = None,
                                lpf_size: int = 5, sobel_kernel_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
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
    :return: the filtered image
    """
    img_height, img_width = img.shape
    adapted_height, adapted_width = img_height, img_width
    filtered_image = np.zeros(img.shape, dtype=img.dtype)
    # theta_map = {}
    phi_map = {}
    ridge_orientation_map = {}

    if gabor_kernel_size is None:
        gabor_kernel_size = block_size

    # partial derivatives
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

    direction_map = np.zeros(shape=(adapted_height, adapted_width))

    for i in inclusive_range(block_size//2, img_height, block_size):
        for j in inclusive_range(block_size//2, img_width, block_size):
            neighborhood_coordinates, neighborhood_shape = neighbor_coordinates(seed_coordinates=(i, j),
                                                                                kernel_size=block_size,
                                                                                height=img_height, width=img_width)

            # block direction estimation
            image_block = np.array([img[px[0], px[1]] for px in neighborhood_coordinates]).reshape(neighborhood_shape)
            block_gx = np.array([gx[px[0], px[1]] for px in neighborhood_coordinates]).reshape(neighborhood_shape)
            block_gy = np.array([gy[px[0], px[1]] for px in neighborhood_coordinates]).reshape(neighborhood_shape)

            vx = np.sum(2*np.multiply(block_gx, block_gy))
            vy = np.sum(np.multiply(block_gx**2, block_gy**2))
            theta_block = 0.5*np.arctan(np.divide(vy, vx+1e-6))  # scalar
            # theta_map[(i//block_size, j//block_size)] = theta_block # leave it here, just in case

            # smoothing & improving orientation approximation
            phi_map[i, j] = (np.cos(2*theta_block), np.sin(2*theta_block))  # (phi_x, phi_y)

            # convert to continuous vector field, by expanding each phi_block
            phi_x_block = phi_map[i, j][0]*np.ones(neighborhood_shape)
            phi_y_block = phi_map[i, j][1]*np.ones(neighborhood_shape)

            # low pass filtering with a median kernel, unit integral
            lpf_phi_x_block = cv2.blur(phi_x_block, (lpf_size, lpf_size))  # phi'_x
            lpf_phi_y_block = cv2.blur(phi_y_block, (lpf_size, lpf_size))  # phi'_y

            # save the current orientation to the ridge map (dictionary)
            current_block_ridge_orientation = 0.5*np.arctan(np.divide(lpf_phi_y_block, lpf_phi_x_block))[0, 0]
            ridge_orientation_map[(i, j)] = current_block_ridge_orientation
            print(current_block_ridge_orientation)

            # Gabor filtering according to estimated ridge block orientation
            gabor_kernel = cv2.getGaborKernel((block_size, block_size), 1.0, current_block_ridge_orientation, 10.0, 1,
                                              ktype=cv2.CV_32F)
            gabor_filtered_block = cv2.filter2D(image_block, -1, gabor_kernel). \
                reshape(image_block.shape[0]*image_block.shape[1], )

            # set the computed pixels in the destination image
            for index, px in enumerate(gabor_filtered_block):
                current_coordinates = neighborhood_coordinates[index]
                filtered_image[current_coordinates[0], current_coordinates[1]] = px

    # reconvert direction_map to np.array
    for coord, ridge_orientation in ridge_orientation_map.items():
        direction_map[coord[0], coord[1]] = ridge_orientation
    return filtered_image, direction_map


def clean(img):
    height, width = img.shape
    new = np.zeros(img.shape, img.dtype)
    for i in range(height):
        for j in range(width):
            new[i][j] = img[i][j]
            if new[i][j] == 1:
                count = 0
                for h in range(max(0, i-2), min(height, i+3)):
                    for k in range(max(0, j - 2), min(width, j + 3)):
                        if img[h][k] == 1:
                            count += 1
                if count <= 2:
                    new[i][j] = 0
    return new


def skeleton_enhancement(img):
    tmp = clean(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel, iterations=1)
    skel = ridge_thinning(closing)
    return skel


def pre_processing(img):
    fingerprint = cv2.bitwise_not(img)
    equalized = cv2.equalizeHist(fingerprint)
    gabor_filtered, block_direction_map = gabor_filtering(img=equalized, block_size=16)
    binarized = otsu(gabor_filtered)
    thinned = ridge_thinning(binarized)
    cleaned = clean(thinned)
    enhanced = skeleton_enhancement(cleaned)
    return enhanced


def find_lines(img):
    height, width = img.shape
    label_map = {}
    label = 1
    for i in range(height):
        for j in range(width):
            if img[i][j] == 1 and (i, j) not in label_map:
                label_map[(i, j)] = label
                label_line(i, j, label_map, label, height, width)
                label += 1
    return label_map, label


def label_line(i, j, label_map, label, height, width):
    for h in range(max(0, i - 1), min(height, i + 2)):
        for k in range(max(0, j - 1), min(width, j + 2)):
            if img[h][k] == 1 and (h, k) not in label_map:
                label_map[(h, k)] = label
                label_line(h, k, label_map, label, height, width)


def print_fingerprint_lines(label_map, label):
    height, width = img.shape
    blank = np.zeros((height, width, 3), np.uint8)
    current = 0
    for l in range(1, label+1):
        pos = [k for k, v in label_map.items() if v == l]
        if(current != l):
            current = l
            r = np.random.randint(50, 255)
            g = np.random.randint(50, 255)
            b = np.random.randint(50, 255)
        for (i, j) in pos:
         blank[i][j] = (b, g, r)
    print_color_image(blank)


if __name__ == '__main__':
    fingerprint = load_image(filename="test4.jpg",
                             cv2_read_param=0)
    # display_image(img=fingerprint, cmap="gray", title="Original fingerprint")

    img = pre_processing(fingerprint)

    label_map, label = find_lines(img)
    print_fingerprint_lines(label_map, label)
    gabor_filtered, ridge_map = gabor_filtering_block_level(img=equalized)
    display_image(img=gabor_filtered, title="Gabor filtering")
    print("End")
