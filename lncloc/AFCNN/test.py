import numpy as np

from util import create_model

x = np.random.random((3,7228))
model = create_model(x)
model.load_weights('weights.best.hdf5')
print(model.predict(x))
