import numpy as np
from scipy.stats import binom
from scipy.io import loadmat

data = np.load('../data/8mer.npy')
datanor = np.load('../data/8mernor.npy')
# data = loadmat(r'D:\matlab\bin\5mer.mat')['lncRNA5mer655']
# datanor = loadmat(r'D:\matlab\bin\5mernor.mat')['lnc5mer655nor']

# m_c = np.sum(data[:426])
# m_e = np.sum(data[426:456])
# m_n = np.sum(data[456:612])
# m_r = np.sum(data[612:])


m_n = np.sum(data[:156])
m_c = np.sum(data[156:582])
m_r = np.sum(data[582:625])
m_e = np.sum(data[625:])

M = m_c + m_e + m_n + m_r

q_n = m_n / M
q_c = m_c / M
q_r = m_r / M
q_e = m_e / M
Q = [q_c, q_e, q_n, q_r]

# ni_c = np.sum(data[:426],axis=0)
# ni_e = np.sum(data[426:456],axis=0)
# ni_n = np.sum(data[456:612],axis=0)
# ni_r = np.sum(data[612:],axis=0)

ni_n = np.sum(data[:156], axis=0)
ni_c = np.sum(data[156:582], axis=0)
ni_r = np.sum(data[582:625], axis=0)
ni_e = np.sum(data[625:], axis=0)

W = [ni_c, ni_e, ni_n, ni_r]
W = np.array(W)
W = W.T
G = np.sum(data, axis=0)
PP = []
fea_len = data.shape[1]
for i in range(fea_len):
    print(i, '正在进行')
    P = []
    for j in range(4):
        sum = 0
        for k in np.arange(W[i][j], G[i] + 1):
            sum += binom.pmf(k, G[i], Q[j])
        P.append(sum)
    PP.append(P)
PP = np.array(PP)
CL = 1 - PP
max_CL = np.max(CL, axis=1)
max_CL = max_CL.reshape(1, max_CL.shape[0]).T
print(max_CL)
index = np.argmax(CL, axis=1)
index = index.reshape(1, index.shape[0]).T
Cli = np.hstack((max_CL, index))
Climax = Cli[:, 0]
Climax = Climax.reshape(1, Climax.shape[0]).T
print(Climax)
Feorder = np.arange(0, fea_len).reshape(1, fea_len).T

CLimax_oder = np.hstack((Climax, Feorder))
CLimax_oder = CLimax_oder[np.argsort(-CLimax_oder[:, 0])]
CLoder8 = CLimax_oder[:, 1]
lnc8mernor655CL = []
for i in range(fea_len):
    print(i, '正在进行')
    E = datanor[:, int(CLoder8[i])]
    E = E.reshape(len(E), 1)
    if i == 0:
        lnc8mernor655CL = E
    else:
        lnc8mernor655CL = np.c_[lnc8mernor655CL, E]

np.save(r'../data/8merbino', arr=lnc8mernor655CL)
