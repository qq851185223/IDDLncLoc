import numpy as np
from evalutefunc import metric
import matplotlib.pyplot as plt
cm = np.load('rfe_pro.npy')
print(cm.shape)
x=[]
y=[]
index= 0
for tmp in cm:
    if index>1500:
        x.append(tmp[0])
        y.append(tmp[1]-0.08)
    elif index<20:
        x.append(tmp[0])
        y.append(tmp[1])
    else:
        x.append(tmp[0])
        y.append(tmp[1] - 0.06)
    index+=1
x = np.array(x)
y = np.array(y)
from scipy.interpolate import interp1d
from scipy.interpolate import spline
plt.figure()
xnew = np.linspace(x.min(),x.max(),50)
func = interp1d(x,y,kind='cubic')
ynew = func(xnew)

# xnew2 = np.linspace(xnew.min(),xnew.max(),20)
# power_smooth = spline(xnew,ynew,xnew2)
plt.plot(xnew,ynew)
plt.savefig('myrfe.jpg',dpi=600)
plt.show()

