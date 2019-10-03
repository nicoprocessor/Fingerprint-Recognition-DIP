#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Gabor filtering utility functions"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import cv2
import scipy.ndimage as ndimage
import scipy.signal as signal
from fingerprint_enhancement import normalize
from typing import Tuple, Union

import utils


def gabor_filter(image: np.ndarray, orientation_map: np.ndarray, frequency_map: np.ndarray,
                 block_size: int = 32, gabor_kernel_size: int = 16) -> np.ndarray:
    """
    Filter the image with a bank of Gabor kernel, modulated according the orientation and frequency map.
    :param image: the original image
    :param orientation_map: the ridge orientation map
    :param frequency_map: the ridge estimated frequency map
    :param block_size: the window size
    :param gabor_kernel_size: the size of the Gabor kernel
    :return: the filtered image
    """
    filtered_img = np.zeros(image.shape)
    height, width = image.shape
    i = 0
    j = 0
    while i < height:
        while j < width:
            # extract current block
            img_block = np.zeros((block_size, block_size), dtype=image.dtype)
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    img_block[h-i, k-j] = image[h, k]

            frequency_block = frequency_map[i, j]

            # frequency threshold
            if frequency_block < 0.0:
                filtered_block = img_block
            else:
                kernel = get_gabor_kernel(gabor_kernel_size, orientation_map[i, j], frequency_map[i, j])
                filtered_block = cv2.filter2D(img_block, -1, kernel)

            # set dst pixels
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    filtered_img[h, k] = filtered_block[h-i, k-j]
            j += block_size
        i += block_size
        j = 0
    return filtered_img


def get_frequency_map(img: np.ndarray, block_size: int = 32, improved_freq: bool = False) -> np.ndarray:
    """
    Estimate the frequency of ridges patterns of each block
    :param img: the original image
    :param block_size: the size of the window
    :param improved_freq: improve the frequencies estimation
    :return: n image with the same size of the original, where in each cell
    is stored the main frequency  of the ridges that belong to that block
    """
    frequency_map = np.empty(img.shape)
    height, width = img.shape
    i = 0
    j = 0
    while i < height:
        while j < width:
            img_block = np.zeros((block_size, block_size), dtype=img.dtype)
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    img_block[h-i][k-j] = img[h][k]

            # calculate the FFT of the current block
            tmp_frequency = np.fft.fft2(img_block)
            tmp_shift = np.fft.fftshift(tmp_frequency)

            umax = 0
            vmax = 0
            fmax = 0
            for u in range(block_size//2):
                for v in range(block_size):
                    f = np.abs(tmp_shift[u][v])
                    if f > fmax:
                        fmax = f
                        umax = u
                        vmax = v
            if improved_freq:
                if np.abs(tmp_shift[umax][vmax-1]) > np.abs(tmp_shift[umax][vmax+1]):
                    new_u = umax-(np.abs(tmp_shift[umax][vmax-1])/(
                            np.abs(tmp_shift[umax][vmax-1])+np.abs(tmp_shift[umax][vmax])))
                else:
                    new_u = umax+(np.abs(tmp_shift[umax][vmax+1])/(
                            np.abs(tmp_shift[umax][vmax+1])+np.abs(tmp_shift[umax][vmax])))

                if np.abs(tmp_shift[umax-1][vmax]) > np.abs(tmp_shift[umax+1][vmax]):
                    new_v = vmax-(np.abs(tmp_shift[umax-1][vmax])/(
                            np.abs(tmp_shift[umax-1][vmax])+np.abs(tmp_shift[umax][vmax])))
                else:
                    new_v = vmax+(np.abs(tmp_shift[umax+1][vmax])/(
                            np.abs(tmp_shift[umax+1][vmax])+np.abs(tmp_shift[umax][vmax])))
                fij = np.sqrt((new_u-block_size//2)**2+(new_v-block_size//2)**2)/block_size
            else:
                fij = np.sqrt((umax-block_size//2)**2+(vmax-block_size//2)**2)/block_size
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    frequency_map[h][k] = fij
            j += block_size
        i += block_size
        j = 0
    return frequency_map


def quantize_val(val, val_list):
    v = 0
    tmp = 1e10
    for i in range(val_list.size):
        if np.abs(val - val_list[i]) < tmp:
            v = val_list[i]
            tmp = np.abs(val - val_list[i])
    return v


def get_orientation_map(image: np.ndarray, block_size: int = 16, sobel_kernel_size: int = 3) -> np.ndarray:
    """
    Estimate the orientation of the ridges in each block
    :param image: the original image
    :param block_size: the size of the window
    :param sobel_kernel_size: the size of the Sobel kernel
    :return: an image with the same size of the original, where in each cell
    is stored the orientation of the ridges that belong to that block
    """
    height, width = image.shape
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    i = 0
    j = 0
    orientation_coherence = np.zeros(shape=(height, width), dtype=float)
    orientation = np.zeros(shape=(height, width), dtype=float)

    while i < height:
        while j < width:
            gxy, gyy, gxx = 0, 0, 0
            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    gxy += gx[h][k]*gy[h][k]
                    gxx += gx[h][k]**2
                    gyy += gy[h][k]**2
            theta = np.rad2deg(0.5*np.arctan2(2*gxy, gxx-gyy))
            theta = quantize_val(theta, np.arange(-180, 180, 5))
            coherence = np.sqrt((gxx-gyy)**2+4*(gxy**2))/(gxx+gyy+1e-6)

            for h in range(i, min(i+block_size, height)):
                for k in range(j, min(j+block_size, width)):
                    orientation_coherence[h, k] = coherence
                    orientation[h, k] = theta
            j += block_size
        i += block_size
        j = 0
    return orientation


def get_gabor_kernel(kernel_size: int, orientation: float, frequency: float) -> np.ndarray:
    """
    Compute the Gabor kernel.
    :param kernel_size: the size of the kernel
    :param orientation: the orientation angle of the kernel
    :param frequency: the frequency of the modulating sine wave
    :return: the Gabor kernel
    """
    x_sigma = 3
    y_sigma = 3
    orientation = np.deg2rad(orientation)
    kernel = np.empty((kernel_size, kernel_size))
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i-kernel_size//2
            y = j-kernel_size//2
            theta = np.pi/2-orientation
            x_theta = x*np.cos(theta)+y*np.sin(theta)
            y_theta = -x*np.sin(theta)+y*np.cos(theta)
            kernel[i, j] = np.exp(-((x_theta**2)/(x_sigma**2)+(y_theta**2)/(y_sigma**2))/2)*np.cos(
                2*np.pi*frequency*x_theta)
    return kernel


def rotate_crop(image, angle):
    """
    Rotate an image and crop the result so that there are no black borders.
    This implementation is based on this stackoverflow answer:
        http://stackoverflow.com/a/16778797
    :param image: the image to rotate.
    :param angle: the angle in radians.
    :returns: the rotated and cropped image.
    """

    h, w = image.shape

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.0*sin_a*cos_a*side_long:
        # half constrained case: two crop corners touch the longer side,
        # the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a-sin_a*sin_a
        wr, hr = (w*cos_a-h*sin_a)/cos_2a, (h*cos_a-w*sin_a)/cos_2a

    image = ndimage.interpolation.rotate(image, np.degrees(angle), reshape=False)

    hr, wr = int(hr), int(wr)
    y, x = (h-hr)//2, (w-wr)//2
    return image[y:y+hr, x:x+wr]


def direction_regularization(ridge_orientation: np.ndarray, coherence_map: np.ndarray) -> np.ndarray:
    """
    Perform angle regularization on the ridge orientation map
    :param ridge_orientation: the ridges orientations
    :param coherence_map: the map containing orientation coherence
    :return: the regularized map
    """
    ridge_height, ridge_width = ridge_orientation.shape
    regularized_map = np.zeros(shape=(ridge_height, ridge_width, 2), dtype=ridge_orientation.dtype)

    for i in range(ridge_height):
        for j in range(ridge_width):
            regularized_direction = np.array([coherence_map[i, j]*np.cos(2*ridge_orientation[i, j]),
                                              coherence_map[i, j]*np.sin(2*ridge_orientation[i, j])])
            regularized_map[i, j] = regularized_direction
    return regularized_map


def display_orientation_map(ridge_orientation: np.ndarray, coherence_map: np.ndarray,
                            block_size: int = 16, regularize=False) -> np.ndarray:
    """
    Displays the vector field defined by the ridge orientation matrix
    :param ridge_orientation: the ridge orientation matrix
    :param coherence_map: the map containing orientation coherence
    :param block_size: the size of each block. By default it is set to 16
    :param regularize: if True performs ridge direction regularization
    :return:
    """
    ridges_height, ridges_width = ridge_orientation.shape
    result_image = np.zeros(ridge_orientation.shape)
    regularized_orientation_map = np.zeros((ridges_height, ridges_width, 2), dtype=ridge_orientation.dtype)

    # direction regularization
    if regularize:
        regularized_orientation_map = direction_regularization(ridge_orientation, coherence_map)

    for i in utils.inclusive_range(block_size//2, ridges_height, block_size):
        for j in utils.inclusive_range(block_size//2, ridges_width, block_size):
            seed = [i, j]  # origin of the current block
            kernel_coordinates, kernel_shape = utils.get_neighbor_coordinates(seed_coordinates=seed,
                                                                              kernel_size=block_size,
                                                                              height=ridges_height,
                                                                              width=ridges_width)
            if regularize:
                block_regularized_direction_x = np.array(
                    [regularized_orientation_map[px[0], px[1]][0] for px in kernel_coordinates])
                block_regularized_direction_y = np.array(
                    [regularized_orientation_map[px[0], px[1]][1] for px in kernel_coordinates])

                current_orientation = [np.mean(block_regularized_direction_x),
                                       np.mean(block_regularized_direction_y)]

                # Rotate 90 degrees anticlockwise
                # current_orientation = rotate_vector(vec=current_orientation, theta=np.radians(90))
            else:
                current_angle = ridge_orientation[i, j]
                # Rotate 90 degrees anticlockwise and convert to vector
                current_orientation = [np.cos(ridge_orientation[i, j]+np.pi*0.5),
                                       np.sin(ridge_orientation[i, j]+np.pi*0.5)]

            normalized_orientation = current_orientation/np.linalg.norm(current_orientation)
            rescaling_factor = block_size//3
            translation_vector = rescaling_factor*normalized_orientation

            # Compute anchor points
            anchor_top = utils.translate_vector(vec=np.array(seed), translation_vector=translation_vector).astype(int)
            anchor_bottom = utils.symmetrical_wrt_center_point(symmetry_origin=np.array(seed), point=anchor_top)

            # draw the line
            cv2.arrowedLine(result_image, tuple(anchor_top), tuple(anchor_bottom), (255, 255, 255), 1)
    return result_image

# def fail_gabor_filtering(img: np.ndarray, block_size: int = 16,
#                          gabor_kernel_size: Union[int, None] = None,
#                          lpf_size: int = 5, sobel_kernel_size: int = 5,
#                          precise_orientation_map: bool = True) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Estimates the direction of each ridge and furrows using Hung least squares approximation
#     and discard background blocks
#     :param img: the original image
#     :param block_size: the size of the block. By default it is set at 16.
#     :param gabor_kernel_size: the size of the Gabor kernel used to extract local features.
#                 By default it is set to be equal to the block size.
#     :param lpf_size: the size of the low pass filter kernel used to improve the orientation ridge.
#                 By default it is set at 5.
#     :param sobel_kernel_size: the size of the Sobel kernel used to extract partial derivatives.
#                 By default it is set at 5.
#     :param precise_orientation_map: if True (default) uses a more precise algorithm for ridge direction estimation.
#     :return: the filtered image
#     """
#     height, width = img.shape
#     filtered_image = np.zeros(img.shape, dtype=img.dtype)
#     orientation_map = {}
#     phi_map = {}
#     improved_orientation_map = {}
#
#     if gabor_kernel_size is None:
#         gabor_kernel_size = block_size
#
#     # partial derivatives
#     gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
#     gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
#
#     direction_map = np.zeros(shape=(int(np.ceil(height/block_size)), int(np.ceil(width/block_size))))
#
#     for i in inclusive_range(block_size//2, height, block_size):
#         for j in inclusive_range(block_size//2, width, block_size):
#             neighborhood_coordinates, neighborhood_shape = neighbor_coordinates(seed_coordinates=(i, j),
#                                                                                 kernel_size=block_size,
#                                                                                 height=height, width=width)
#
#             # block direction estimation
#             image_block = np.array([img[px[0], px[1]] for px in neighborhood_coordinates]).reshape(neighborhood_shape)
#             block_gx = np.array([gx[px[0], px[1]] for px in neighborhood_coordinates]).reshape(neighborhood_shape)
#             block_gy = np.array([gy[px[0], px[1]] for px in neighborhood_coordinates]).reshape(neighborhood_shape)
#
#             vx = np.sum(2*np.multiply(block_gx, block_gy))
#             vy = np.sum(np.subtract(block_gx**2, block_gy**2))
#
#             # TODO careful here
#             # block_orientation = 0.5*np.arctan(np.divide(vy, vx+1e-6))  # scalar
#             block_orientation = 0.5*np.arctan2(vy, vx)  # scalar
#
#             # Rotate the orientations so that they point along the ridges, and wrap
#             # them into only half of the circle (all should be less than 180 degrees).
#             orientation_map[(i, j)] = block_orientation
#
#             if precise_orientation_map:  # smoothing & improving orientation estimation
#                 phi_map[i, j] = (np.cos(2*block_orientation), np.sin(2*block_orientation))  # (phi_x, phi_y)
#
#                 # convert to continuous vector field, by expanding each phi_block
#                 phi_x_block = phi_map[i, j][0]*np.ones(neighborhood_shape)
#                 phi_y_block = phi_map[i, j][1]*np.ones(neighborhood_shape)
#
#                 # low pass filtering with a median kernel, unit integral
#                 lpf_phi_x_block = cv2.blur(phi_x_block, (lpf_size, lpf_size))  # phi'_x
#                 lpf_phi_y_block = cv2.blur(phi_y_block, (lpf_size, lpf_size))  # phi'_y
#
#                 # save the current orientation to the ridge map (dictionary)
#                 improved_block_orientation = 0.5*np.arctan(np.divide(lpf_phi_y_block, lpf_phi_x_block))[0, 0]
#                 improved_orientation_map[(i, j)] = improved_block_orientation
#             else:
#                 improved_block_orientation = block_orientation
#                 improved_orientation_map = orientation_map
#
#             # Gabor filtering according to estimated ridge block orientation
#             kernel = get_gabor_kernel(kernel_size=block_size, orientation=improved_block_orientation, frequency=1/9)
#
#             gabor_filtered_block = cv2.filter2D(image_block, -1, kernel). \
#                 reshape(image_block.shape[0]*image_block.shape[1], )
#
#             # set the computed pixels in the destination image
#             for index, px in enumerate(gabor_filtered_block):
#                 current_coordinates = neighborhood_coordinates[index]
#                 filtered_image[current_coordinates[0], current_coordinates[1]] = px
#
#         # reconvert direction_map to np.array
#         for coord, ridge_orientation in improved_orientation_map.items():
#             direction_map[coord[0]//block_size, coord[1]//block_size] = ridge_orientation
#     return filtered_image, direction_map


# def estimate_frequencies(image, orientations, w=32):
#     """
#     Estimate ridge or line frequencies in an image, given an orientation field.
#     This is more or less an implementation of of the algorithm in Chapter 2.5 in
#     the paper:
#         Fingerprint image enhancement: Algorithm and performance evaluation
#         Hong, L., Wan, Y. & Jain, A. (1998)
#     :param image: The image to estimate orientations in.
#     :param orientations: An orientation field such as one returned from the
#                          estimateOrientations() function.
#     :param w: The block size.
#     :returns: An ndarray the same shape as the image, filled with frequencies.
#     """
#     height, width = image.shape
#     y_blocks, x_blocks = height//w, width//w
#     F = np.empty((y_blocks, x_blocks))
#
#     for y in range(y_blocks):
#         for x in range(x_blocks):
#             orientation = orientations[y*w+w//2, x*w+w//2]
#
#             block = image[y*w:(y+1)*w, x*w:(x+1)*w]
#             block = rotate_crop(block, np.pi*0.5+orientation)
#             if block.size == 0:
#                 F[y, x] = -1
#                 continue
#
#             columns = np.sum(block, (0,))
#             columns = normalize(columns)
#             peaks = signal.find_peaks_cwt(columns, np.array([3]))
#             if len(peaks) < 2:
#                 F[y, x] = -1
#             else:
#                 f = (peaks[-1]-peaks[0])/(len(peaks)-1)
#                 if f < 5 or f > 15:
#                     F[y, x] = -1
#                 else:
#                     F[y, x] = 1/f
#
#     frequencies = np.full(image.shape, -1.0)
#     F = np.pad(F, 1, mode="edge")
#     for y in range(y_blocks):
#         for x in range(x_blocks):
#             surrounding = F[y:y+3, x:x+3]
#             surrounding = surrounding[np.where(surrounding >= 0.0)]
#             if surrounding.size == 0:
#                 frequencies[y*w:(y+1)*w, x*w:(x+1)*w] = -1
#             else:
#                 frequencies[y*w:(y+1)*w, x*w:(x+1)*w] = np.median(surrounding)
#     return frequencies

# Orientated window solution
# def estimate_frequencies(img: np.ndarray, orientation_map: np.ndarray, w: int = 32) -> np.ndarray:
#     """
#
#     :param img:
#     :param w:
#     :param orientation_map:
#     :return:
#     """
#
#     height, width = img.shape
#     i = 0
#     j = 0
#     while i < height:
#         while j < width:
#             x_signature = np.zeros(shape=(w,))
#             tmp = np.zeros((w, w), dtype=img.dtype)
#             for h in range(i, min(i+w, height)):
#                 for k in range(j, min(j+w, width)):
#                     tmp[h-i][k-j] = img[h][k]
#
#             # oriented window
#             roi = rotate_crop(image=tmp, angle=orientation_map[i, j]+np.pi*0.5)
#
#             for k in range(w):
#                 seed = ((i+min(i+w, height)//2), (j+min(j+w, width)//2))
#                 for d in range(w):
#                     u = seed[0]+(d-w//2)*np.cos(orientation_map[seed[0], seed[1]]) \
#                         +(k-w//2)*np.sin(orientation_map[seed[0], seed[1]])
#
#                     v = seed[1]+(d-w//2)*np.sin(orientation_map[seed[0], seed[1]]) \
#                         +(w//2-k)*np.cos(orientation_map[seed[0], seed[1]])
#
#             j += w
#         i += w
#         j = 0
