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


def match(skel1, I1, skel2, I2):
    n = 10
    L = 5  # use inter ridge length
    for i in range(len(I1)):
        for j in range(len(I2)):
            minutia1 = I1[i]
            minutia2 = I2[j]
            sample1 = sample(skel1, minutia1, L, n)
            sample2 = sample(skel2, minutia2, L, n)
            # sim = similarity(sample1, sample2)
            # print(sim)
            # if sim >= 0.8:
            #     minutiae.print_minutiae(skel1, sample1, 255, 0, 0)
            #     minutiae.print_minutiae(skel2, sample2, 255, 0, 0)
            #     print('ok')
            minutiae_match()


n_count = 0


def sample(skel, minutia, L, n):
    height, width = skel.shape
    i, j, _ = minutia
    samples = []
    processed = []
    samples.append((i, j))
    processed.append((i, j))
    #minutiae.print_minutiae(skel, [(i, j)], 0, 0, 255)
    global n_count
    n_count = n
    if minutiae.count_pixels(skel, i, j) == 2:
        n_count = n-1
        sample_r(skel, height, width, samples, processed, i, j, L, L)
        # minutiae.print_minutiae(skel, samples, 0, 0, 255)
    if minutiae.count_pixels(skel, i, j) == 4:
        for h in range(max(0, i-1), min(height, i+2)):
            for k in range(max(0, j-1), min(width, j+2)):
                if skel[h][k] == 1:
                    n_count = n//3
                    sample_r(skel, height, width, samples, processed, i, j, L, L)
       # minutiae.print_minutiae(skel, samples, 255, 0, 0)
    return samples


def sample_r(skel, height, width, samples, processed, i, j, L, L_reset):
    global n_count
    if n_count > 0:
        for h in range(max(0, i-1), min(height, i+2)):
            for k in range(max(0, j-1), min(width, j+2)):
                if skel[h][k] == 1:
                    if (h, k) not in processed:
                        if L == 0:
                            n_count = n_count - 1
                            samples.append((h, k))
                            processed.append((h, k))
                            sample_r(skel, height, width, samples, processed, h, k, L_reset, L_reset)
                        else:
                            processed.append((h, k))
                            sample_r(skel, height, width, samples, processed, h, k, L-1, L_reset)


# ?????
def similarity(samples1, samples2):
    m = min(len(samples1), len(samples2))
    num = 0
    den = 0
    for i in range(m):
        x1, _, _ = samples1[i]
        x2, _, _ = samples2[i]
        num += x1*x2
    for i in range(m):
        x1, _, _ = samples1[i]
        x2, _, _ = samples2[i]
        den += (x1**2)*(x2**2)
    return num/np.sqrt(den)


#not tested
def minutiae_transform(x, y, theta, minutiae):
    for i in range(len(minutiae)):
        xi, yi, thetai = minutiae[i]
        matrix = np.array([np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1])
        res = np.dot(matrix, np.array([xi-x],[yi-y],[thetai - theta]))
        minutiae[i] = res #vedere tipo di output


#not tested
def minutiae_match(I1, I2, r0, theta0):
    mm_tot = 0
    for i in range(len(I1)):
        for j in range(len(I2)):
            xi, yi, thetai = I1[i]
            xj, yj, thetaj = I1[j]
            sd = np.sqrt(((xi-xj)**2)+((yi-yj)**2))
            dd = np.min(np.abs(thetai-thetaj), 360 - np.abs(thetai-thetaj))
            if sd < r0 and dd < theta0:
                mm_tot += 1
    return mm_tot/(np.max(len(I1), len(I2))+1)# ci va il +1?