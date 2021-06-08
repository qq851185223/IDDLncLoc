import numpy as np
from scipy import interp
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat

# x = np.load(r'C:\Users\Amber\Desktop\feature\8merbino.npy')[:,:6554]
# #x2= loadmat(r'C:\Users\Amber\Desktop\feature\8merbino.mat')['kmerCLorder'][:,:6554]
# print(x.shape)
# x = MinMaxScaler((0,1)).fit_transform(x)
# y = np.genfromtxt(r'C:\Users\Amber\Desktop\feature\label.txt')
#
# rfe = RFE(LinearSVC(),n_features_to_select=4000)
# x =rfe.fit_transform(x,y)
# print(x.shape)

from Bio import SeqIO
from Bio import pairwise2 as pw2

first_fasta = r'C:\Users\Amber\Desktop\r_right.fasta'
# second_fasta = r'C:\Users\Amber\Desktop\e_wrong_c.fasta'
second_fasta = r'C:\Users\Amber\Desktop\e_right.fasta'
first_dict = SeqIO.to_dict(SeqIO.parse(open(first_fasta), 'fasta'))  # 直接转为字典格式
second_dict = SeqIO.to_dict(SeqIO.parse(open(second_fasta), 'fasta'))
res = []
index = 0
for t in first_dict:
    # 11
    t_len = len(first_dict[t].seq)
    same = []
    for t2 in second_dict:
        try:
            global_align = pw2.align.globalxx(first_dict[t].seq, second_dict[t2].seq)
            matched = global_align[0][2]
            percent_match = (matched / t_len) * 100
            # print(t + '\t' + t2 + '\t' + str(percent_match) + '\n')
            same.append(percent_match)
        except:
            continue
    same = np.array(same)
    print(same.mean())
    res.append(same.mean())
    index += 1
res = np.array(res)
print(res.mean())

# data = np.genfromtxt(r'C:\Users\Amber\Desktop\tmp.txt')
# print(data.mean())
