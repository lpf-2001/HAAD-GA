import numpy as np

data = np.load('awf2.npz')
print("包含的键:", data.files)

x_test = data['data']
y_test = data['labels']

print("x_test shape:", x_test.shape)
print("y_test shape:", np.unique(y_test))