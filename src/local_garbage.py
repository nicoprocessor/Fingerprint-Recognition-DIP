#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Main module"""
from numpy.core.multiarray import ndarray

__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
from skimage.filters import gabor_kernel
import logging
from skimage.morphology import skeletonize
from typing import Tuple, Union
import cv2
import scipy.ndimage as ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt
import utils
from utils import load_image
from utils import neighbor_coordinates
from utils import display_image
from utils import inclusive_range
from utils import print_images
from utils import print_color_image


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


def get_orientation_map(img, w=32):
    height, width = img.shape
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    Gx = cv2.filter2D(img, -1, sobel_x)
    Gy = cv2.filter2D(img, -1, sobel_y)
    i = 0
    j = 0
    orientation = np.zeros((height, width), dtype=float)
    while i < height:
        while j < width:
            vx = 0
            vy = 0
            for h in range(i, min(i+w, height)):
                for k in range(j, min(j+w, width)):
                    vx += 2 * Gx[h][k] * Gy[h][k]
                    vy += (Gx[h][k]**2 - Gy[h][k]**2)
            val = 0.5*np.arctan2(vx, vy)
            for h in range(i, min(i+w, height)):
                for k in range(j, min(j+w, width)):
                    orientation[h][k] = (val + np.pi * 0.5) % np.pi
            j += w
        i += w
        j = 0
    return orientation


def get_frequency_map(img, w=12):
    freq = np.empty(img.shape)
    height, width = img.shape
    i = 0
    j = 0
    while i < height:
        while j < width:
            tmp = np.zeros((w, w), dtype=img.dtype)
            for h in range(i, min(i+w, height)):
                for k in range(j, min(j+w, width)):
                    tmp[h - i][k - j] = img[h][k]
            tmp_frequency = np.fft.fft2(tmp)
            tmp_shift = np.fft.fftshift(tmp_frequency)
            umax = 0
            vmax = 0
            fmax = 0
            for u in range(w//2):
                for v in range(w):
                    f = np.abs(tmp_shift[u][v])
                    if f > fmax:
                        fmax = f
                        umax = u
                        vmax = v
            print(str((umax, vmax)))
            fij = np.sqrt((umax - w//2)**2 + (vmax - w//2)**2)/w
            for h in range(i, min(i + w, height)):
                for k in range(j, min(j + w, width)):
                    freq[h][k] = fij
            j += w
        i += w
        j = 0
    return freq


def get_gabor_kernel(size, angle, frequency):
    angle += np.pi * 0.5
    cos = np.cos(angle)
    sin = -np.sin(angle)

    yangle = lambda x, y: x * cos + y * sin
    xangle = lambda x, y: -x * sin + y * cos

    xsigma = ysigma = 4

    kernel = np.empty((size, size))
    for i in range(0, size):
        for j in range(0, size):
            x = i - size / 2
            y = j - size / 2
            kernel[i, j] = np.exp(-((xangle(x, y) ** 2) / (xsigma ** 2) + (yangle(x, y) ** 2) / (ysigma ** 2)) / 2) * np.cos(2 * np.pi * frequency * xangle(x, y))
    return kernel


def rotateAndCrop(image, angle):
    """
    Rotate an image and crop the result so that there are no black borders.
    This implementation is based on this stackoverflow answer:
        http://stackoverflow.com/a/16778797
    :param image: The image to rotate.
    :param angle: The angle in gradians.
    :returns: The rotated and cropped image.
    """

    h, w = image.shape

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long:
        # half constrained case: two crop corners touch the longer side,
        # the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    image = ndimage.interpolation.rotate(image, np.degrees(angle), reshape=False)

    hr, wr = int(hr), int(wr)
    y, x = (h - hr) // 2, (w - wr) // 2

    return image[y:y+hr, x:x+wr]


def normalize(image):
    image = image.astype(float)
    image = np.copy(image)
    image -= np.min(image)
    m = np.max(image)
    if m > 0.0:
        image *= 1.0 / m
    return image


def findMask(image, threshold=0.1, w=32):
    """
    Create a mask image consisting of only 0's and 1's. The areas containing
    1's represent the areas that look interesting to us, meaning that they
    contain a good variety of color values.
    """

    mask = np.empty(image.shape)
    height, width = image.shape
    for y in range(0, height, w):
        for x in range(0, width, w):
            block = image[y:y+w, x:x+w]
            standardDeviation = np.std(block)
            if standardDeviation < threshold:
                mask[y:y+w, x:x+w] = 0.0
            elif block.shape != (w, w):
                mask[y:y+w, x:x+w] = 0.0
            else:
                mask[y:y+w, x:x+w] = 1.0

    return mask


def estimateFrequencies(image, orientations, w=32):
    """
    Estimate ridge or line frequencies in an image, given an orientation field.
    This is more or less an implementation of of the algorithm in Chapter 2.5 in
    the paper:
        Fingerprint image enhancement: Algorithm and performance evaluation
        Hong, L., Wan, Y. & Jain, A. (1998)
    :param image: The image to estimate orientations in.
    :param orientations: An orientation field such as one returned from the
                         estimateOrientations() function.
    :param w: The block size.
    :returns: An ndarray the same shape as the image, filled with frequencies.
    """

    height, width = image.shape
    yblocks, xblocks = height // w, width // w
    F = np.empty((yblocks, xblocks))
    for y in range(yblocks):
        for x in range(xblocks):
            orientation = orientations[y*w+w//2, x*w+w//2]

            block = image[y*w:(y+1)*w, x*w:(x+1)*w]
            block = rotateAndCrop(block, np.pi * 0.5 + orientation)
            if block.size == 0:
                F[y, x] = -1
                continue

            columns = np.sum(block, (0,))
            columns = normalize(columns)
            peaks = signal.find_peaks_cwt(columns, np.array([3]))
            if len(peaks) < 2:
                F[y, x] = -1
            else:
                f = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
                if f < 5 or f > 15:
                    F[y, x] = -1
                else:
                    F[y, x] = 1 / f

    frequencies = np.full(image.shape, -1.0)
    F = np.pad(F, 1, mode="edge")
    for y in range(yblocks):
        for x in range(xblocks):
            surrounding = F[y:y+3, x:x+3]
            surrounding = surrounding[np.where(surrounding >= 0.0)]
            if surrounding.size == 0:
                frequencies[y*w:(y+1)*w, x*w:(x+1)*w] = -1
            else:
                frequencies[y*w:(y+1)*w, x*w:(x+1)*w] = np.median(surrounding)

    return frequencies


def gabor_filter(img, O, F, w=32):
    new = np.zeros(img.shape)
    height, width = img.shape
    i = 0
    j = 0
    while i < height:
        while j < width:
            tmp = np.zeros((w, w), dtype=img.dtype)
            for h in range(i, min(i+w, height)):
                for k in range(j, min(j+w, width)):
                    tmp[h - i][k - j] = img[h][k]
            f = F[i][j]
            if f < 0.0:
                new_tmp = tmp
            else:
                kernel = get_gabor_kernel(16, O[i][j], F[i][j])
                new_tmp = cv2.filter2D(tmp, -1, kernel)
            for h in range(i, min(i+w, height)):
                for k in range(j, min(j+w, width)):
                    new[h][k] = new_tmp[h-i][k-j]
            j += w
        i += w
        j = 0
    return new


def binarization(img: np.ndarray) -> np.ndarray:
    """
    Image binarization
    :param img: the original image
    :return: the binarized image
    """
    block_size = 32
    height, width = img.shape
    i = j = 0
    new = np.zeros(img.shape)
    while i < height:
        while j < width:
            tmp = np.zeros((block_size, block_size), dtype=img.dtype)
            count = 0
            pixel_sum = 0
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    tmp[h-i][k-j] = img[h][k]
                    count += 1
                    pixel_sum += img[h][k]
            mean = pixel_sum/count
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    if tmp[h-i][k-j] > 1.5*mean:
                        new[h][k] = 1
                    else:
                        new[h][k] = 0
            j += block_size
        i += block_size
        j = 0
    return new


#TODO IMPROVE PARAMETERS
def pre_processing(img):
    """
    Preprocessing operations on the image
    :param img: the original image
    :return: the processed image
    """
    negated = cv2.bitwise_not(img)
    denoised = cv2.fastNlMeansDenoising(negated, None, 15)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)
    O = get_orientation_map(normalize(equalized))
    print('ok')
    F = get_frequency_map(normalize(equalized))
    mask = findMask(normalize(equalized))
    image = gabor_filter(normalize(equalized), O, F)
    image = np.where(mask == 1.0, image, 1.0)
    print_images([image, ridge_thinning(binarization(image))])



if __name__ == '__main__':
    fingerprint = load_image(filename="test4.jpg", cv2_read_param=0)

    processed_img = pre_processing(fingerprint)

    # label_map, labels = find_lines(processed_img)
    # ridges = find_ridges(processed_img, label_map, labels)
    # bifurcation = find_bifurcation(processed_img, label_map, labels)
    # print_minutia(processed_img, ridges, 0, 0, 255)
    # print_minutia(processed_img, bifurcation, 255, 0, 0)

