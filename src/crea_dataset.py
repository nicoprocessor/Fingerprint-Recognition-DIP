import utils
import os
from fingerprint_recognition import pre_processing
from minutiae import crossing_numbers
from minutiae import false_minutiae_removal
from minutiae import remove_minutiae
import cv2
from utils import load_image
from minutiae import find_lines
import numpy as np


if __name__ == '__main__':

    folder = '../res'

    for file in os.listdir(folder):
        fingerprint = load_image(filename=file, cv2_read_param=0)
        processed_img, ridge_orientation_map, ridge_frequency = pre_processing(fingerprint)

        minutiae = crossing_numbers(processed_img, ridge_orientation_map)
        minutiae_tuned = minutiae.copy()
        ridge_identification_map, labels = find_lines(processed_img)
        freq = 1 / np.mean(ridge_frequency)

        minutiae_tuned = false_minutiae_removal(processed_img, minutiae_tuned, ridge_identification_map, freq / 1.5)
        minutiae_tuned_removed = remove_minutiae(minutiae_tuned)
        minutiae_normal = false_minutiae_removal(processed_img, minutiae, ridge_identification_map, freq)
        minutiae_normal_removed = remove_minutiae(minutiae_normal)
        utils.save((minutiae_normal_removed, minutiae_tuned_removed), '../res/minutiae_'+file[:-4])
