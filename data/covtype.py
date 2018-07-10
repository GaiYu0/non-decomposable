import numpy as np

xy = np.loadtxt('covtype.data', delimiter=',', dtype=np.int)
x, y = xy[:, :54], xy[:, 54]
np.save('x', x)
np.save('y', y)
