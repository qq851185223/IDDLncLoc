from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
xlabel = ['nucleus','cytoplasm','ribosome','exosome']

cm = np.load('confuse_matrix.npy')
confusion = cm
print(cm)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, xlabel)
plt.yticks(indices, xlabel)
plt.colorbar()
plt.title('The confusion matrix of lncLoc')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[second_index][first_index])
plt.savefig('my_cm.jpg',dpi=700)
plt.show()