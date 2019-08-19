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


def alignment(skel1, I1, skel2, I2):
    n = 10
    L = 5 #use inter ridge length
    minutia1 = I1[39] # for testing
    #minutia2 = I2[39]
    sample(skel1, minutia1, L, n)


def sample(skel, minutia, L, n):
    height, width = skel.shape
    i, j = minutia
    samples = []
    processed = []
    samples.append((i, j))
    processed.append((i, j))
    minutiae.print_minutiae(skel, [(i, j)], 0, 0, 255)
    if minutiae.count_pixels(skel, i, j) == 2:
        sample_r(skel, height, width, samples, processed, i, j, L, n-1, L)
        minutiae.print_minutiae(skel, samples, 0, 0, 255)
    if minutiae.count_pixels(skel, i, j) == 4:
        for h in range(max(0, i - 1), min(height, i + 2)):
            for k in range(max(0, j - 1), min(width, j + 2)):
                if skel[h][k] == 1:
                    sample_r(skel, height, width, samples, processed, i, j, L, n//3, L)
        minutiae.print_minutiae(skel, samples, 255, 0, 0)


def sample_r(skel, height, width, samples, processed, i, j, L, n, L_reset):
    if n > 0:
        for h in range(max(0, i - 1), min(height, i + 2)):
            for k in range(max(0, j - 1), min(width, j + 2)):
                if skel[h][k] == 1:
                    if (h, k) not in processed:
                        if L == 0:
                            samples.append((h, k))
                            processed.append((h, k))
                            sample_r(skel, height, width, samples, processed, h, k, L_reset, n-1, L_reset)
                        else:
                            processed.append((h, k))
                            sample_r(skel, height, width, samples, processed, h, k, L-1, n, L_reset)

# not tested
def similarity(samples1, samples2):
    m = min(len(samples1), len(samples2))
    num = 0
    den = 0
    for i in range(m):
        x1, _ = samples1(i)
        x2, _ = samples2(i)
        num += x1*x2
    for i in range(m):
        x1, _ = samples1(i)
        x2, _ = samples2(i)
        den += (x1**2)*(x2**2)
    return np.sqrt(num/den)
