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


def match(skel1, I1, skel2, I2):
    minutia1 = I1[0]
    x, y, c, o, _ = minutia1
    minutiae_transform(x, y, o, I1)
    minutiae_transform(x, y, o, I2)
    res = minutiae_match(I1, I2, r0=10, theta0=30)
    a = 5


def minutiae_transform(x, y, theta, minutiae):
    for i in range(len(minutiae)):
        xi, yi, c, thetai, validity = minutiae[i]
        matrix = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        m2 = [[xi-x], [yi-y], [thetai - theta]]
        res = np.dot(matrix, m2)
        xnew = res[0][0]
        ynew = res[1][0]
        theta_new = res[2][0]
        minutiae[i] = xnew, ynew, c, theta_new, validity


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