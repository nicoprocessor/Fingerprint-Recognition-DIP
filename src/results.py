#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Results module"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"
import numpy as np
import math
import random
import re
import time
from pathlib import Path
from typing import List, Tuple, Any
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from fingerprint_recognition import *
from utils import save, frange
import sys

Minutia = Tuple[Any, Any, Any, Any, Any]

# global paths
dataset_path = Path.cwd().parent/'res'/'casia'
minutiae_path = Path.cwd().parent/'res'/'minutiae_dataset'
results_path = Path.cwd().parent/'res'/'dst'
global_seed = 101
random.seed(global_seed)


def process_individual_fingerprint_given_path(fingerprint_id: str):
    """
    Process and save a specific fingerprint
    :param fingerprint_id: the fingerprint identifier in the dataset
    """
    patient_number = fingerprint_id[:3]
    side = fingerprint_id[4]
    fingerprint_path = dataset_path/patient_number/side/fingerprint_id
    # fingerprint_id = re.sub(r'.bmp', '', str(fingerprint_path.name))
    print("Processing this: "+fingerprint_id)

    fingerprint = load_image(filename=str(fingerprint_path)+'.bmp', cv2_read_param=0)
    minutiae_normal, minutiae_tuned = process_fingerprint(fingerprint)
    save((fingerprint_id, minutiae_normal, minutiae_tuned),
         '../res/minutiae_dataset/'+fingerprint_id)
    # print("Saved correctly!")


def process_individual_fingerprint(patient_number, side, finger_number, sample_number):
    """
    Process and save a specific fingerprint
    :param patient_number: the number of the patient
    :param side: the hand side
    :param finger_number: the number of the finger
    :param sample_number: the fingerprint sample
    """
    fingerprint_id = '_'.join(
        [str(patient_number).zfill(3), str(side).upper()+str(finger_number), str(sample_number)+'.bmp'])
    fingerprint_path = dataset_path/patient_number/side/fingerprint_id

    fingerprint_id = re.sub(r'.bmp', '', str(fingerprint_path.name))
    fingerprint = load_image(filename=str(fingerprint_path), cv2_read_param=0)
    minutiae_normal, minutiae_tuned = process_fingerprint(fingerprint)

    save((fingerprint_id, minutiae_normal, minutiae_tuned),
         '../res/minutiae_dataset/'+fingerprint_id)


def process_fingerprint(fingerprint: np.ndarray) -> Tuple[List[Minutia], List[Minutia]]:
    """
    The entire minutiae extraction chain
    :param: the fingerprint image
    :return: the minutiae list
    """
    processed_img, ridge_orientation_map, ridge_frequency = pre_processing(fingerprint)
    minutiae = crossing_numbers(processed_img, ridge_orientation_map)
    minutiae_tuned = minutiae.copy()
    ridge_identification_map, labels = find_lines(processed_img)
    freq = 1/np.mean(ridge_frequency)

    minutiae_tuned = false_minutiae_removal(processed_img, minutiae_tuned, ridge_identification_map, freq*2)
    minutiae_tuned_removed = remove_minutiae(minutiae_tuned)
    minutiae_normal = false_minutiae_removal(processed_img, minutiae, ridge_identification_map, freq*2)
    minutiae_normal_removed = remove_minutiae(minutiae_normal)

    #print_minutiae(processed_img, minutiae_tuned_removed, 255, 0, 0, 'tuned')
    #print_minutiae(processed_img, minutiae_normal_removed, 255, 0, 0, 'normal')
    return minutiae_normal_removed, minutiae_tuned_removed


def sample_fingerprint_dataset(n):
    """
    Choose n fingerprint randomly from the whole dataset,
    process them and save the result as a pickle dump reusable file
    :param n: the size of the sampled dataset
    """
    fingerprint_dataset_paths = [p for p in dataset_path.rglob('*') if p.suffix == '.bmp']
    print("Original dataset size: "+str(len(fingerprint_dataset_paths)))
    # pprint.pprint(fingerprint_dataset_paths)

    for k in range(n):
        s = random.sample(fingerprint_dataset_paths, 1)[0]
        fingerprint_id = re.sub(r'.bmp', '', str(s.name))
        s = str(s)

        for sample_number in range(5):
            s_sample = s[:-5]
            s_sample += str(sample_number)+".bmp"
            fingerprint_id_sample = fingerprint_id[:7]
            fingerprint_id_sample += str(sample_number)
            fingerprint = load_image(filename=str(s_sample), cv2_read_param=0)

            print(str(k)+"."+str(sample_number)+") Processing file: "+fingerprint_id_sample)
            minutiae_normal, minutiae_tuned = process_fingerprint(fingerprint)
            save((fingerprint_id_sample, minutiae_normal, minutiae_tuned),
                 '../res/minutiae_dataset/'+fingerprint_id_sample)


def positive_negative_split_sample(size: int, positives_percentage: float = 0.5) -> \
        List[Tuple[Minutia, Minutia]]:
    """

    :param size:
    :param positives_percentage:
    :return:
    """
    testset = []
    neg_samples_1, neg_samples_2 = [], []
    pos_samples_1, pos_samples_2 = [], []
    pos = math.floor(positives_percentage*size)
    dataset_dumps = [p for p in minutiae_path.rglob('*')]

    # positive tests
    # print("{:>12}{:>12}{:>12}".format("Sample #1", "Sample #2", "Expected"))
    p = random.sample(dataset_dumps, pos)

    for k in range(pos):
        p1 = p[k]
        selected_sample = int(str(p1)[-1])
        p2 = list(str(p[k]))
        numbers = [0, 1, 2, 3, 4]
        del numbers[selected_sample]
        r = int(random.choice(numbers))
        p2[-1] = str(r)  # change last character
        p2 = Path("".join(p2))

        with open(p2, 'rb') as file_obj:
            p2_minutia = pickle.load(file_obj)
            pos_samples_2.append(p2_minutia)

        with open(p1, 'rb') as file_obj:
            p1_minutia = pickle.load(file_obj)
            pos_samples_1.append(p1_minutia)

        # print("{:>12}{:>12}{:>9}".format(str(p1_minutia[0]), str(p2_minutia[0]), "Y"))
    testset += zip([x for x in pos_samples_1], [x for x in pos_samples_2])

    # negative tests
    for x in range(size-pos):
        cnt = 0
        s2, s1 = 0, 0
        while s2 == s1:  # retry sampling
            s1 = random.sample(dataset_dumps, 1)[0]
            s2 = random.sample(dataset_dumps, 1)[0]
            cnt += 1
            if cnt == 10:
                raise Exception("Impossible dataset sampling!")

        with open(s1, 'rb') as file_obj:
            s1_minutia = pickle.load(file_obj)
            neg_samples_1.append(s1_minutia)

        with open(s2, 'rb') as file_obj:
            s2_minutia = pickle.load(file_obj)
            neg_samples_2.append(s2_minutia)

        # print("{:>12}{:>12}{:>9}".format(str(s1_minutia[0]), str(s2_minutia[0]), "N"))
    testset += zip(neg_samples_1, neg_samples_2)
    return testset


def eval_performance(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    h = hamming(y_pred, y_true)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("Hamming distance: {}\n"
          "Precision: {}\n"
          "Recall: {}\n"
          "F1 score: {}\n"
          "Confusion Matrix:\n{}".format(h, precision, recall, f_score, cm))


def plot_roc(y_true, y_pred, threshold):
    """

    :param y_true:
    :param y_pred:
    :param threshold:
    :return:
    """
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = {0:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example, threshold: {}'.format(threshold))
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def load_results(thresh, positive_perc, test_size, random_seed):
    filepath = results_path/"{}_{}_{}_{}".format(int(thresh*10), int(positive_perc*10), test_size, random_seed)
    print("Loading: "+str(filepath))

    with open(filepath, 'rb') as file:
        res = pickle.load(file)
    return res


# def eval_predictions(y_true, scores: np.ndarray):
#     """
#     Eval scores according to threshold
#     :param y_true: expected classes
#     :param scores: matching scores
#     """
#     y_pred = []
#     for th in frange(start=0.4, stop=0.8, step=0.1):
#         for i in range(len(scores)):
#             if scores[i] >= th:
#                 y_pred[i] =


if __name__ == '__main__':
    # Call this when you want to process new fingerprints (it may take some time)
    sys.setrecursionlimit(5000)
    #sample_fingerprint_dataset(100)
    load_res = False

    # fingerprint matching
    threshold = 0.2
    positive_percentage = 0.5
    test_set_size = 100

    if not load_res:
        test_set = positive_negative_split_sample(size=test_set_size, positives_percentage=positive_percentage)
        positives = math.floor(positive_percentage*test_set_size)
        y_true = np.array([1]*positives+[0]*(len(test_set)-positives), dtype=np.uint8)
        scores = np.zeros(y_true.shape, dtype=np.float16)
        y_pred = np.zeros(y_true.shape, dtype=np.uint8)

        start = time.perf_counter()
        for k, test in enumerate(test_set):
            id_1, m_1, m_tuned_1 = test[0][0], test[0][1], test[0][2]
            id_2, m_2, m_tuned_2 = test[1][0], test[1][1], test[1][2]

            score = matching.match(m_1, m_2, m_tuned_1, m_tuned_2)
            scores[k] = score
            if score >= threshold:
                y_pred[k] = 1

            print("{:<4} - {:>12}{:>12}{:>9}{:>9}{:>12}".format(k, id_1, id_2, str(y_true[k]),
                                                                str(y_pred[k]), str(round(score, 3))))

        # save results
        dst_path = results_path/"{}_{}_{}_{}".format(int(threshold*10), int(positive_percentage*10), test_set_size,
                                                     global_seed)
        save(obj=(y_true, y_pred), name=str(dst_path))

        # Execution performance
        end = time.perf_counter()
        elapsed = end-start
        print("\nTotal execution time: {:.3}s\n"
              "Match execution time: {:4f}s/match".format(elapsed, (elapsed/test_set_size)))
        veri = int(len(scores)*positive_percentage)
        print('media veri:' + str(np.mean(scores[:veri])))
        print('var veri:' + str(np.var(scores[:veri])))
        print('75 percentile veri:' + str(np.percentile(scores[:veri], 75)))
        print('media falsi:' + str(np.mean(scores[veri:])))
        print('var falsi:' + str(np.var(scores[veri:])))
        print('75 percentile falsi:' + str(np.percentile(scores[veri:], 75)))
    else:
        # Load previously computed result
        y_true, y_pred = load_results(thresh=threshold, positive_perc=positive_percentage,
                                      test_size=test_set_size, random_seed=global_seed)

    # performance evaluation
    eval_performance(y_true, y_pred)
    plot_roc(y_true, y_pred, threshold)  # TODO fix

    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0
    #
    # for i in range(len(y_true)):
    #     if y_true[i] == y_pred[i] == 1:
    #         tp+=1
    #     if y_true[i] == y_pred[i] == 0:
    #         tn+=1
    #     if y_true[i] == 1 and y_pred[i] == 0:
    #         fn+=1
    #     if y_true[i] == 0 and y_pred[i] == 1:
    #         fp+=1
    #
    # print('precision =' + str(tp/(tp+fp)))
    # print('recall =' + str(tp / (tp + fn)))
    # print('accuracy =' + str((tp+tn) / (fn + tn + tp + fp)))
    # print('debug')

