from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np


x = torch.ones(1, 1, requires_grad=True)
y = x + 2

z = y * y

z.backward()

x = [1, 2, 3]

print(x[1:3])
