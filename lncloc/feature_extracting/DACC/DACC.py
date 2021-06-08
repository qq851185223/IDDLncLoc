#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy as np

from feature_extracting.DACC.descnucleotide import *
from feature_extracting.DACC.pubscripts import *


def get_DACC():
    my_property_name, my_property_value = check_parameters.check_acc_arguments()
    file = '../../data/input.fasta'
    fastas = read_fasta_sequences.read_nucleotide_sequences(file)

    encodings = ACC.make_acc_vector(fastas, my_property_name, my_property_value, 2, 2)

    print(encodings)
    np.save('../../data/DACC', arr=encodings)
    print('完成DACC特征的提取')


if __name__ == '__main__':
    get_DACC()
