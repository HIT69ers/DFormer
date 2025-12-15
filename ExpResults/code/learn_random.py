import torch
import random
import numpy as np


# np.random.seed(10)
# print(np.random.rand(12))
# # np.random.seed(10)
# print(np.random.rand(12))

# random.seed(10)
# print(random.random())
# # random.seed(10)
# print(random.random())
# # random.seed(10)
# print(random.random())

manual_seed = 10
torch.manual_seed(manual_seed)
print(torch.rand(6))
torch.manual_seed(manual_seed)
print(torch.rand(6))
print(torch.rand(6))
torch.manual_seed(manual_seed)
print(torch.rand(12))