import matplotlib.pyplot as plt
import numpy as np

N = 2  # size of mini batch
H = 3  # dimension number of hidden vector
T = 20  # length of time series data

dh = np.ones((N, H))
np.random.seed(3)  # set seed of random number for simulate
Wh = np.random.randn(H, H)
# Wh = np.random.randn(H, H) * 0.5

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh ** 2)) / N
    norm_list.append(norm)

print(norm_list)

plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('Time Step')
plt.ylabel('Norm')
plt.show()
