#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Utility functions for fingerprint minutiae analysis"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import numpy as np
import cv2
import minutiae
from utils import print_images
from utils import print_color_image
from utils import rotate_vector

def minutiae_match(I1, I2, r0, theta0):
    mm_tot = 0
    for i in range(len(I1)):
        for j in range(len(I2)):
            xi, yi, ci, thetai, _ = I1[i]
            xj, yj, cj, thetaj, _ = I2[j]
            sd = np.sqrt(((xi-xj)**2)+((yi-yj)**2))
            dd = min(np.abs(thetai-thetaj), 360-np.abs(thetai-thetaj))
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
    print("minutiale alignment")
    scalings = np.array([0.8, 0.95, 1, 1.05, 1.1])
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
                            for hh in range(max(h-1, 0), min(h+2, deltas_x.size)):
                                for kk in range(max(k-1, 0), min(k+2, deltas_y.size)):
                                    if hh == h and kk == k:
                                        accumulator[j][i][hh][kk] += 3
                                    accumulator[j][i][hh][kk] += 1
    scale, theta, dx, dy = np.unravel_index(np.argmax(accumulator, axis=None), accumulator.shape)
    return scalings[scale], thetas[theta], deltas_x[dx], deltas_y[dy]


def match_hough(img, I1, I2, I11, I22):
    scale, theta, deltax, deltay = hough_match(I1, I2)
    # print(scale, theta, deltax, deltay)
    minutiae.print_minutiae3(img, I11, I22, "before alignment")
    print("minutiale transformation")
    minutiae_transform_hough(scale, theta, deltax, deltay, I22)
    # minutiae_transform_hough(scale, theta, deltax, deltay, I2)
    minutiae.print_minutiae3(img, I11, I22, "after alignment")
    res = minutiae_match_hough(I11, I22, r0=20, theta0=15)
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
    print("calculating score")
    mm_tot = 0
    for i in range(len(I1)):
        for j in range(len(I2)):
            yi, xi, ci, thetai, _ = I1[i]
            yj, xj, cj, thetaj, _ = I2[j]
            sd = np.sqrt(((xi-xj)**2)+((yi-yj)**2))
            dd = min(np.abs(thetai-thetaj), 360 - np.abs(thetai-thetaj))
            dd = min(np.abs(thetai-thetaj), 360-np.abs(thetai-thetaj))
            if sd < r0 and dd < theta0 and ci == cj:
                mm_tot += 1
    res = mm_tot/(max(len(I1), len(I2)))
    if res > 0.3:
        return "The given fingerprints match!"
    else:
        return "The given fingerprints don't match!"
