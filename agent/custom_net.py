from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

class GoofyAhhNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.input = nn.Linear(3,512)
        self.linear1 = nn.Linear(512,256)
        self.linear2 = nn.Linear(256,128)
        self.linear3 = nn.Linear(128,64)
        self.output = nn.Linear(64,4)
  def forward(self,x):
        x = self.input(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        return x