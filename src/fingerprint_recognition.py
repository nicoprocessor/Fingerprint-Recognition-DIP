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
import cv2

from utils import load_image
from utils import neighbor_coordinates
from utils import display_image
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



def gabor_filtering(img: np.ndarray, block_size: int) -> np.ndarray:
    """
    Estimates the direction of each ridge and furrows using Hung least squares approximation
    and discard background blocks
    :param img: the original image
    :param block_size: the size of the block
    :return: the filtered image
    """
    sobel_kernel_size = 5
    img_height, img_width = img.shape
    adapted_height, adapted_width = img_height, img_width
    filtered_image = np.zeros(img.shape, dtype=img.dtype)
    theta_map = {}

    # partial derivatives
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

    # adapt size
    if img_height%block_size == 0:
        adapted_height = img_height+block_size
    if img_width%block_size == 0:
        adapted_width = img_width+block_size

    direction_map = np.zeros(shape=(adapted_height, adapted_width))

    for i in range(0, adapted_height, block_size):
        for j in range(0, adapted_width, block_size):
            block_coordinates, neighborhood_shape = neighbor_coordinates(seed_coordinates=(i, j),
                                                                         kernel_size=block_size,
                                                                         height=img_height, width=img_width)

            # block direction estimation
            block = np.array([fingerprint[px[0], px[1]] for px in block_coordinates]).reshape(neighborhood_shape)
            block_gx = np.array([gx[px[0], px[1]] for px in block_coordinates]).reshape(neighborhood_shape)
            block_gy = np.array([gy[px[0], px[1]] for px in block_coordinates]).reshape(neighborhood_shape)

            vx = np.sum(2*np.multiply(block_gx, block_gy))
            vy = np.sum(np.multiply(block_gx**2, block_gy**2))

            theta_block = 0.5*np.arctan(np.divide(vy, vx+1e-6))
            theta_map[(i//block_size, j//block_size)] = theta_block

            # filtering
            g_kernel = cv2.getGaborKernel((block_size, block_size), 1.0, theta_block, 10.0, 1, ktype=cv2.CV_32F)
            filtered_block = cv2.filter2D(block, cv2.CV_8UC3, g_kernel).reshape(block.shape[0]*block.shape[1], )

            for index, px in enumerate(filtered_block):
                current_coordinates = block_coordinates[index]
                filtered_image[current_coordinates[0], current_coordinates[1]] = px

    # reconvert theta_map to np.array
    for coord, theta in theta_map.items():
        direction_map[coord[0], coord[1]] = theta
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
    label_map = np.zeros(img.shape, np.uint8)
    label = 1
    for i in range(height):
        for j in range(width):
            if img[i][j] == 1 and label_map[i][j] == 0:
                label_map[i][j] = label
                label_line(i, j, label_map, label, height, width)
                label += 1
    return label_map, label


def label_line(i, j, label_map, label, height, width):
    for h in range(max(0, i - 1), min(height, i + 2)):
        for k in range(max(0, j - 1), min(width, j + 2)):
            if img[h][k] == 1 and label_map[h][k] == 0:
                label_map[h][k] = label
                label_line(h, k, label_map, label, height, width)


## Very slow, use only for test
def print_fingerprint_lines(label_map, label):
    height, width = img.shape
    blank = np.zeros((height, width, 3), np.uint8)
    for l in range(1, label+1):
        r = np.random.randint(50, 255)
        g = np.random.randint(50, 255)
        b = np.random.randint(50, 255)
        for i in range(height):
            for j in range(width):
                if label_map[i][j] == l:
                    blank[i][j] = (b, g, r)
    print_color_image(blank)


if __name__ == '__main__':
    fingerprint = load_image(filename="test4.jpg",
                             cv2_read_param=0)
    # display_image(img=fingerprint, cmap="gray", title="Original fingerprint")

    img = pre_processing(fingerprint)

    label_map, label = find_lines(img)
    print_fingerprint_lines(label_map, label)
