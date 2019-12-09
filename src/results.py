#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Results module"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import pandas as pd
import re
import pickle
import random
import math
import pprint
from pathlib import Path
from utils import load_image
from utils import save
from fingerprint_recognition import *
from typing import List, Tuple, Any

Minutia = Tuple[Any, Any, Any, Any, Any]


def process_fingerprint_manually():
    while True:
        src_path = Path.cwd().parent/'res'/'CASIA-Fingerprint'
        patient_number = str(input("Enter the patient number: ")).zfill(3)
        side = str(input("Enter the hand side [L/R]: ")).upper()
        finger_number = str(input("Enter the finger number [0-3]: "))
        sample_number = str(input("Enter the fingerprint sample number[0-4]: "))
        fingerprint_id = '_'.join([patient_number, side+finger_number, sample_number+'.bmp'])
        fingerprint_path = src_path/patient_number/side/fingerprint_id
        print(fingerprint_path)

        fingerprint_id = re.sub(r'.bmp', '', str(fingerprint_path.name))
        fingerprint = load_image(filename=str(fingerprint_path), cv2_read_param=0)
        print("Processing file: "+fingerprint_id)
        minutiae_normal, minutiae_tuned = process_fingerprint(fingerprint)

        save((fingerprint_id, minutiae_normal, minutiae_tuned),
             '../res/minutiae_dataset/'+fingerprint_id)
        print("Minutiae saved succesfully!")

        end = str(input("Continue?[Y(Enter)/N]: "))
        if end == "N":
            return


def process_fingerprint(fingerprint: np.ndarray) -> Tuple[List[Minutia], List[Minutia]]:
    """The entire minutiae extraction chain
    :param: the fingerprint image
    :return: the minutiae list
    """
    processed_img, ridge_orientation_map, ridge_frequency = pre_processing(fingerprint)
    minutiae = crossing_numbers(processed_img, ridge_orientation_map)
    minutiae_tuned = minutiae.copy()
    ridge_identification_map, labels = find_lines(processed_img)
    freq = 1/np.mean(ridge_frequency)

    minutiae_tuned = false_minutiae_removal(processed_img, minutiae_tuned, ridge_identification_map, freq/1.5)
    minutiae_tuned_removed = remove_minutiae(minutiae_tuned)
    minutiae_normal = false_minutiae_removal(processed_img, minutiae, ridge_identification_map, freq)
    minutiae_normal_removed = remove_minutiae(minutiae_normal)
    return minutiae_normal_removed, minutiae_tuned_removed


def sample_fingerprint_dataset(n):
    """
    Choose n fingerprint randomly from the whole dataset,
    process them and save the result as a pickle dump reusable file
    :param n: the size of the sampled dataset
    """
    src_path = Path.cwd().parent/'res'/'CASIA-Fingerprint'
    # dst_path = Path.cwd().parent/'res'/'minutiae_dataset'
    fingerprint_dataset_paths = [p for p in src_path.rglob('*') if p.suffix == '.bmp']
    print("Original dataset size: "+str(len(fingerprint_dataset_paths)))
    # pprint.pprint(fingerprint_dataset_paths)

    for k in range(n):
        s = random.sample(fingerprint_dataset_paths, 1)[0]
        fingerprint_id = re.sub(r'.bmp', '', str(s.name))
        fingerprint = load_image(filename=str(s), cv2_read_param=0)
        print(str(k)+") Processing file: "+fingerprint_id)
        minutiae_normal, minutiae_tuned = process_fingerprint(fingerprint)

        save((fingerprint_id, minutiae_normal, minutiae_tuned),
             '../res/minutiae_dataset/'+fingerprint_id)


def positive_negative_sample(dataset: List[Tuple[Minutia, Minutia]], size: int, positives_percentage: float = 0.1) -> \
        List[Tuple[Minutia, Minutia]]:
    if len(dataset) < size:
        raise ValueError("Required size is too big for the actual dataset!")
    data = []
    test, expected = [], []
    positives = math.floor(positives_percentage*size/100)

    # positive tests
    s = random.sample(dataset, positives)

    # negative tests
    for x in range(size-positives):
        cnt = 0
        e, t = 0, 0
        while e == t:
            t = random.sample(dataset, 1)
            e = random.sample(dataset, 1)
            cnt += 1
            if cnt == 10:
                raise Exception("Impossible dataset sampling!")
        test += t
        expected += e
    data += zip(test, expected)
    return data


# def get_results(predicted: List[int], actual: List[int]):
#     data = {predicted, actual}
#     df = pd.DataFrame(data, columns=['Actual', 'Predicted'])
#     cm = pd.crosstab(df['Actual'], df['Predicted'],
#                      rownames=['Actual'], colnames=['Predicted'])
#     print(cm)
#     print(df)


if __name__ == '__main__':
    # Processa e salva una singola impronta manualmente
    process_fingerprint_manually()

    # Processa e salva "n" impronte alla volta, casualmente
    # sample_fingerprint_dataset(200)
