#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""Results module"""
__author__ = "Nicola Onofri, Luigi Bonassi"
__license__ = "GPL"
__email__ = "nicola.onofri@gmail.com, " \
            "l.bonassi005@studenti.unibs.it"

import pandas as pd
from typing import List


def get_results(predicted: List[int], actual: List[int]):
    data = {predicted, actual}
    df = pd.DataFrame(data, columns=['Actual', 'Predicted'])
    cm = pd.crosstab(df['Actual'], df['Predicted'],
                     rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    print(df)
