#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import random
class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.epsilon = epsilon
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.initVals = initVals
		self.Qvalues = {}
		# self.policy = {}
		self.cache = {}
		self.visited_states = []

	def learn(self):

		cur = (self.cache['sCur'],self.cache['aCur'])
		nex = (self.cache['sNext'],self.cache['aNext'])
		a   = self.learningRate
		r   = self.cache['rPrev']
		g   = self.discountFactor
		
		tmp = self.Qvalues[cur]
		############### UPDATE Q VALUE ##########################
		if self.cache['sNext'] == None:
			self.Qvalues[cur] += a * (r - self.Qvalues[cur])
		else:
			self.Qvalues[cur] += a * (r + g*self.Qvalues[nex] - self.Qvalues[cur])
		

		return self.Qvalues[cur] - tmp

	def act(self):

		if random.random() < self.epsilon: # e greedy
			action = np.random.choice(self.possibleActions)
		else:
			action_vals = [self.Qvalues[(self.curState,ac_)] for ac_ in self.possibleActions]
			best_action_val = max(action_vals)
			actionIdx = np.random.choice([i for i,ac_ in enumerate(action_vals)  if ac_ == best_action_val])
			action = self.possibleActions[actionIdx]
		return action

	def setState(self, state):
		if state not in self.visited_states:
			self.visited_states.append(state)
			for ac_ in self.possibleActions:
				self.Qvalues[(state,ac_)] = self.initVals
			
		self.curState = state

	def setExperience(self, state, action, reward, status, nextState):
		if nextState not in self.visited_states:
			self.visited_states.append(nextState)
			for ac_ in self.possibleActions:
				self.Qvalues[(nextState,ac_)] = self.initVals

		if not self.cache:	#is empty
			self.cache = {
				'sPrev': None,
				'rPrev': None,
				'sCur' : state,
				'aCur' : None,
				'rCur' : reward,
				'sNext' : nextState,
				'aNext' : action
			}
		else:
			# update cache
			self.cache['sPrev'] = self.cache['sCur']
			self.cache['rPrev'] = self.cache['rCur']
			self.cache['sCur'] = state
			self.cache['aCur'] = self.cache['aNext']
			self.cache['rCur'] = reward
			self.cache['sNext'] = nextState
			self.cache['aNext'] = action
			
		self.status = status
		

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		lr = max(1e-1 - 1e-1*episodeNumber/4500,1e-5)
		e  = max(1 - episodeNumber/4500,1e-6)
		
		return lr,e

	def toStateRepresentation(self, state):
		return tuple(state)

	def reset(self):
		self.cache = {}

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=5000)

	args=parser.parse_args()
	
	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.1, 0.99,0.5)

	# Run training using SARSA
	numTakenActions = 0 
	goals = 0
	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True
	
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			# print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			if status == 1:
				goals+=1
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()
		print(episode ,':',goals)
	
