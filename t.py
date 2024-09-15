import os
print(os.getcwd())
import matplotlib.pyplot as plt
from sym_nn.distributions import TestDistribution

class Args:
    def __init__(self):
        self.min_radius = 1-1e-4
        self.max_radius = 1+1e-4

args = Args()

distribution = TestDistribution(args)

data = distribution.sample(1000, rotate=True)

x = data["positions"]

plt.scatter(x[:, :, 0], x[:, :, 1])
plt.show()