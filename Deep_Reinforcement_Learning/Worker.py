import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random

def hard_copy(targetValueNetwork, valueNetwork):
	for target_param, param in zip(targetValueNetwork.parameters(), valueNetwork.parameters()):
					target_param.data.copy_(param.data)


def train(idx, args, value_network, target_value_network, optimizer, lock, counter):
	port = 8000+ 50*idx 
	env = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=123)
	env.connectToServer()
	do_opt_step = False
	do_hard_copy = False
	
	
	total_reward = 0
	save_counter = 1
	for episode in range(args.numEpisodes):
		
		timestep = 0
		total_reward = 0
		obs = env.reset()
		done = False
		save_model = False
		do_opt_step = False
		do_hard_copy = False
		while not done and timestep<500:
			
			# obs to tensor
			obs_tensor = torch.Tensor(obs).unsqueeze(0)
			# choose action
			act = chooseAction(obs_tensor,value_network,episode,idx)
			# execute action
			act_str = env.possibleActions[act]
			next_obs, rewards, done, status,info = env.step(act_str)
			# update total reward
			total_reward += rewards
			# reward to tensor
			reward_tensor = torch.Tensor([rewards])
			# next obs to tensor
			next_obs_tensor = torch.Tensor(next_obs).unsqueeze(0)
			# update counters and flags
			timestep+=1
			with lock:
				counter.value = counter.value +1
				if timestep % args.aSyncFreq == 0 or done:
					do_opt_step = True
				if counter.value % args.copy_freq == 0:
					do_hard_copy =True
				if counter.value % 1e6 == 0:
					save_model =True
			current_count = counter.value
			#forward pass for our networks
			predicted_vals = computePrediction(obs_tensor,act,value_network)
			target_vals    = computeTargets(reward_tensor, next_obs_tensor, args.gamma, done, target_value_network)
			# loss function calculation
			loss_function = nn.MSELoss()
			err = loss_function(predicted_vals, target_vals)
			# accumulate gradients
			err.backward()

			# update optimizer
			if do_opt_step:
				with lock:
					optimizer.step()
					optimizer.zero_grad()
				do_opt_step = False
			#update global network
			if do_hard_copy:
				with lock:
					hard_copy(target_value_network, value_network)
				do_hard_copy = False
			# update current state
			obs = next_obs

		if save_model:
			#save model
			
			saveModelNetwork(value_network,'params_'+str(save_counter))
			save_counter+=1
			#change learning rate
			change_lr(current_count,optimizer)

	saveModelNetwork(value_network,'params_latest')

	
def change_lr(counter_val,optimizer):
	lr = max(10 ** (-4 - counter_val // 1e6),1e-7) 
	for group in optimizer.param_groups:
		group['lr'] = lr

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	if done:
		return torch.tensor([reward])
	else:
		target_vals = targetNetwork(nextObservation)
		max_tar = torch.max(target_vals[0],0)[0]
		return reward + discountFactor * max_tar
		
def computePrediction(state, action, valueNetwork):
	vals = valueNetwork.forward(state)
	return vals[0,action]

def getEpsilon(idx,episode):
	e_worker = 1 - 0.1 * idx 
	epsilon = max(e_worker - episode/(8000.0*1.0),0)
	return epsilon

def chooseAction(state,valueNetwork,episode,worker_idx):
	vals = valueNetwork.forward(state)
	if random.random() < getEpsilon(worker_idx,episode):  # e - greedy policy
		act = random.randint(0,3)
	else:
		act = torch.max(vals, dim=1)[1].item()
	return act

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)
	




