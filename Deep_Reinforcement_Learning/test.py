import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
#from Networks import ValueNetwork
from SharedAdam import SharedAdam
from Environment import HFOEnv
#from Worker import train
import random

port = random.randint(0,9999)
hfoEnv = HFOEnv(numTeammates=1, numOpponents=1, port=port, seed=123)
hfoEnv.connectToServer()
obs = hfoEnv.reset()
aa = hfoEnv.possibleActions

obsn = hfoEnv.step(aa[0])
obsn = hfoEnv.step(aa[1])[0]
obsn = torch.Tensor(obsn).unsqueeze(0)

obsn = torch.cat((obsn,torch.Tensor([0]).unsqueeze(0)), 1)
print('obs1',obsn,obsn.shape)