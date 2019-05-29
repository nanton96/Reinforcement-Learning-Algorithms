#!/usr/bin/env python3
# encoding utf-8


import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
import numpy as np

class IndependentQLearningAgent(Agent):
	
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()
		self.epsilon = epsilon
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.initVals = initVals
		self.Qvalues = {}
		self.cache = {}
		self.visited_states = []

	def learn(self):

		cur = (self.cache['sCur'],self.cache['aNext'])
		a   = self.learningRate
		r   = self.cache['rCur']
		g   = self.discountFactor

		tmp = self.Qvalues[cur]
		############### UPDATE Q VALUE ##########################
		Qsa =[]
		for ac in self.possibleActions:
			pair = (self.cache['sNext'],ac)
			Qsa.append(self.Qvalues[pair])
		Qmax = max(Qsa)
		self.Qvalues[cur] += a * (r + g*Qmax - self.Qvalues[cur])
		
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
		lr = max(1 - episodeNumber/45000,1e-5)
		e  = max(1 - episodeNumber/45000,1e-6)
		return lr,e
	def toStateRepresentation(self, state):
		if isinstance(state, str):
			return state
		else:
			b = list(state)
			c =[]
			for a in b: 
				c.extend(a)
			d = []
			for a in c:
				d.append(tuple(a))
			state = tuple(d)
			return state

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		
#!/usr/bin/env python3
# encoding utf-8

		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	counter = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0
		
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))

				actions.append(agents[agentIdx].act())
			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			if reward[0]>0:
				counter+=1
			
				
			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation

	
		print('Episode', episode,'goals: ', counter)
