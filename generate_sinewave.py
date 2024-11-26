import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

N = 100 # number of samples
L = 1000 # length of sequence
T = 20 # period of sin wave

x = np.empty((N, L), np.float32)
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1) # shift the wave randomly
y = np.sin(x/1.0/T).astype(np.float32) 



plt.figure(figsize=(10, 8))
plt.title('Sin wave')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(x.shape[1]), y[0,:], 'r', linewidth=2.0)
plt.show()