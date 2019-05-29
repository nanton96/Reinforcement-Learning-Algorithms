#!/usr/bin/env python3
# encoding utf-8


# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from Networks import ValueNetwork
from SharedAdam import SharedAdam
from Environment import HFOEnv
from Worker import train
import random
parser = argparse.ArgumentParser(description='Deep asynchronous RL HFO')
parser.add_argument('--num_processes', type=int, default=8,
                    help='number of different processes to be run')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='optimizer learning rate')
#parser.add_argument('--max_grads',type=float,default=1.0,help = 'max_grads')
parser.add_argument('--gamma',type=float,default=0.999,help='gamma')
parser.add_argument('--copy_freq', type=int, default=10000,
                    help='copy_freq')
parser.add_argument('--aSyncFreq', type=int, default=10,
help='aSyncFreq')
parser.add_argument('--numEpisodes',type=int,default=8000,help='Number of episodes')

if __name__ == "__main__" :
	args = parser.parse_args()
	
	
	value_network = ValueNetwork(15,[60,60,30],4)
	value_network.share_memory()
	target_value_network = ValueNetwork(15,[60,60,30],4)
	target_value_network.share_memory()
	
	print('lr',args.learning_rate)
	optimizer = SharedAdam(params=value_network.parameters(),lr=args.learning_rate)
	optimizer.share_memory()

	counter = mp.Value('i', 0)
	
	lock = mp.Lock()
	processes = []
	for idx in range(0, args.num_processes):
		trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter)
		p = mp.Process(target=train, args=trainingArgs)
		p.start()
		processes.append(p)
	for p in processes:
		p.join()



