#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import thorsen_utils
import scipy.misc as misc
import scipy.ndimage as ndimage

import utils


# ONLY THORSEN CODE


def kernel_from_function(size, f):
    """
    Creates a kernel of the given size, populated with values obtained by
    calling the given function.

    :param size: The desired size of the kernel.
    :param f:    The function.
    :returns:    The created kernel.
    """

    kernel = np.empty((size, size))
    for i in range(0, size):
        for j in range(0, size):
            kernel[i, j] = f(i-size/2, j-size/2)
    return kernel


def gabor_kernel(size, angle, frequency):
    """
    Create a Gabor kernel given a size, angle and frequency.

    Code is taken from https://github.com/rtshadow/biometrics.git
    """

    angle += np.pi*0.5
    cos = np.cos(angle)
    sin = -np.sin(angle)

    y_angle = lambda x, y:x*cos+y*sin
    x_angle = lambda x, y:-x*sin+y*cos

    x_sigma = y_sigma = 4
    return thorsen_utils.kernelFromFunction(size, lambda x, y:np.exp(-(
            (x_angle(x, y)**2)/(x_sigma**2)+
            (y_angle(x, y)**2)/(y_sigma**2))/2)*np.cos(2*np.pi*frequency*x_angle(x, y)))


def gabor_filter(image, orientations, frequencies, w=32):
    result = np.empty(image.shape)

    height, width = image.shape
    for y in range(0, height-w, w):
        for x in range(0, width-w, w):
            orientation = orientations[y+w//2, x+w//2]
            frequency = thorsen_utils.averageFrequency(frequencies[y:y+w, x:x+w])

            if frequency < 0.0:
                result[y:y+w, x:x+w] = image[y:y+w, x:x+w]
                continue

            kernel = gabor_kernel(16, orientation, frequency)
            result[y:y+w, x:x+w] = thorsen_utils.convolve(image, kernel, (y, x), (w, w))

    return thorsen_utils.normalize(result)


def gaborFilterSubdivide(image, orientations, frequencies, rect=None):
    if rect:
        y, x, h, w = rect
    else:
        y, x = 0, 0
        h, w = image.shape

    result = np.empty((h, w))
    orientation, deviation = thorsen_utils.averageOrientation(
        orientations[y:y+h, x:x+w], deviation=True)

    if (deviation < 0.2 and h < 50 and w < 50) or h < 6 or w < 6:
        frequency = thorsen_utils.averageFrequency(frequencies[y:y+h, x:x+w])

        if frequency < 0.0:
            result = image[y:y+h, x:x+w]
        else:
            kernel = gabor_kernel(16, orientation, frequency)
            result = thorsen_utils.convolve(image, kernel, (y, x), (h, w))

    else:
        if h > w:
            hh = h//2
            result[0:hh, 0:w] = gaborFilterSubdivide(image, orientations, frequencies, (y, x, hh, w))
            result[hh:h, 0:w] = gaborFilterSubdivide(image, orientations, frequencies, (y+hh, x, h-hh, w))
        else:
            hw = w//2
            result[0:h, 0:hw] = gaborFilterSubdivide(image, orientations, frequencies, (y, x, h, hw))
            result[0:h, hw:w] = gaborFilterSubdivide(image, orientations, frequencies, (y, x+hw, h, w-hw))

    if w > 20 and h > 20:
        result = thorsen_utils.normalize(result)

    return result


if __name__ == '__main__':
    np.set_printoptions(
        threshold=np.inf,
        precision=4,
        suppress=True)

    print("Reading image")
    image = ndimage.imread(sourceImage, mode="L").astype("float64")
    if options.images > 0:
        thorsen_utils.showImage(image, "original", vmax=255.0)

    print("Normalizing")
    image = thorsen_utils.normalize(image)
    if options.images > 1:
        thorsen_utils.showImage(image, "normalized")

    print("Finding mask")
    mask = thorsen_utils.findMask(image)
    if options.images > 1:
        thorsen_utils.showImage(mask, "mask")

    print("Applying local normalization")
    image = np.where(mask == 1.0, thorsen_utils.localNormalize(image), image)
    if options.images > 1:
        thorsen_utils.showImage(image, "locally normalized")

    print("Estimating orientations")
    orientations = np.where(mask == 1.0, thorsen_utils.estimateOrientations(image), -1.0)
    if options.images > 0:
        thorsen_utils.showOrientations(image, orientations, "orientations", 8)

    print("Estimating frequencies")
    frequencies = np.where(mask == 1.0, thorsen_utils.estimateFrequencies(image, orientations), -1.0)
    if options.images > 1:
        thorsen_utils.showImage(thorsen_utils.normalize(frequencies), "frequencies")

    print("Filtering")
    if options.subdivide:
        image = thorsen_utils.normalize(gaborFilterSubdivide(image, orientations, frequencies))
    else:
        image = gabor_filter(image, orientations, frequencies)
    image = np.where(mask == 1.0, image, 1.0)
    if options.images > 0:
        thorsen_utils.showImage(image, "gabor")

    if options.binarize:
        print("Binarizing")
        image = np.where(mask == 1.0, thorsen_utils.binarize(image), 1.0)
        if options.images > 0:
            thorsen_utils.showImage(image, "binarized")

    if options.images > 0:
        plt.show()

    if not options.dryrun:
        misc.imsave(destinationImage, image)
