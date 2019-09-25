#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Utility functions for fingerprint minutiae analysis"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import minutiae
from utils import print_images
from utils import print_color_image
from utils import rotate_vector


def match(img, I1, I2):
    max_res = 0
    for i in range(len(I1)):
        tmp = np.copy(I2)
        minutia1 = I1[i]
        y, x, c, o, _ = minutia1
        for j in range(len(I2)):
            minutiae_transform(y, x, c, o, tmp, j)
            # minutiae.print_minutiae3(img, I1, tmp)
            res = minutiae_match(I1, I2, r0=25, theta0=30)
            if res > max_res:
                max_res = res
    return max_res


def minutiae_transform(y, x, c, o, minutiae, j):
    yi, xi, ci, oi, validity = minutiae[j]
    dx = xi - x
    dy = yi - y
    do = oi - o
    if c == ci and np.abs(dx) < 25 and np.abs(dy) < 25:
        for i in range(len(minutiae)):
            xi, yi, c, thetai, validity = minutiae[i]
            matrix = np.array([[np.cos(do), -np.sin(do)], [np.sin(do), np.cos(do)]])
            m2 = np.array([[xi-dx], [yi-dy]])
            res = matrix @ m2
            xnew = res[0][0]
            ynew = res[1][0]
            minutiae[i] = int(ynew), int(xnew), c, thetai, validity


# not tested
def minutiae_match(I1, I2, r0, theta0):
    mm_tot = 0
    for i in range(len(I1)):
        for j in range(len(I2)):
            xi, yi, ci, thetai, _ = I1[i]
            xj, yj, cj, thetaj, _ = I2[j]
            sd = np.sqrt(((xi-xj)**2)+((yi-yj)**2))
            dd = min(np.abs(thetai-thetaj), 360 - np.abs(thetai-thetaj))
            if sd < r0 and dd < theta0 and ci == cj:
                mm_tot += 1
    return mm_tot/(max(len(I1), len(I2)))


def quantize(val, val_list):
    q = 0
    tmp = 1e10
    for i in range(val_list.size):
        if np.abs(val - val_list[i]) < tmp:
            q = i
            tmp = np.abs(val - val_list[i])
    return q


def hough_match(set1, set2):
    scalings = np.array([0.95, 1, 1.05])
    thetas = np.arange(-90, 90, 5)
    deltas_x = np.arange(-50, 50, 1)
    deltas_y = np.arange(-50, 50, 1)
    accumulator = np.zeros((scalings.size, thetas.size, deltas_x.size, deltas_y.size))
    for minutiae1 in set1:
        for minutiae2 in set2:
            py, px, cn_p, alpha, _ = minutiae1
            qy, qx, cn_q, beta, _ = minutiae2
            for i in range(thetas.size):
                if int(alpha) + thetas[i] == int(beta) and cn_p == cn_q:
                    print("Attenzione!")
                    for j in range(scalings.size):
                        matrix = np.array([[np.cos(thetas[i]), np.sin(thetas[i])], [-np.sin(thetas[i]), np.cos(thetas[i])]])
                        q = np.array([[qx], [qy]])
                        p = np.array([[px], [py]])
                        res = q - scalings[j] * matrix @ p
                        deltax = res[0][0]
                        deltay = res[1][0]
                        h = quantize(deltax, deltas_x)
                        k = quantize(deltay, deltas_y)
                        if -50 < deltax < 50 and -50 < deltay < 50:
                            accumulator[j][i][h][k] += 1
    scale, theta, dx, dy = np.unravel_index(np.argmax(accumulator, axis=None), accumulator.shape)
    return scalings[scale], thetas[theta], deltas_x[dx], deltas_y[dy]


def match_hough(img, I1, I2):
    scale, theta, deltax, deltay = hough_match(I1, I2)
    minutiae.print_minutiae3(img, I1, I2)
    minutiae_transform_hough(scale, theta, deltax, deltay, I2)
    minutiae.print_minutiae3(img, I1, I2)
    res = minutiae_match_hough(I1, I2, r0=10, theta0=10)
    return res


def minutiae_transform_hough(scale, theta, deltax, deltay, set):
    for i in range(len(set)):
        y, x, cn, alpha, validity = set[i]
        matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        xy = np.array([[x], [y]])
        dxy = np.array([[deltax], [deltay]])
        res = scale*matrix @ xy
        res = res - dxy
        xnew = int(res[0][0])
        ynew = int(res[1][0])
        set[i] = ynew, xnew, cn, alpha, validity


def minutiae_match_hough(I1, I2, r0, theta0):
    mm_tot = 0
    for i in range(len(I1)):
        for j in range(len(I2)):
            yi, xi, ci, thetai, _ = I1[i]
            yj, xj, cj, thetaj, _ = I2[j]
            sd = np.sqrt(((xi-xj)**2)+((yi-yj)**2))
            dd = min(np.abs(thetai-thetaj), 360 - np.abs(thetai-thetaj))
            if sd < r0 and dd < theta0 and ci == cj:
                mm_tot += 1
    return mm_tot

