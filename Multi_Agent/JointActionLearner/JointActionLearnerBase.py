#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import numpy as np	
class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()
		self.epsilon = epsilon
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.initVals = initVals
		self.Qvalues = {}  # Q(s,a1,a2) = {(s1,a1_a,a1_b): q1, .... }
		self.Csa = {}      # C(s,a)     = {(s1,a1_b)     : c1, ...}
		self.n   = {}      # n(s)       = {s1 : n1, ...}
		self.cache = {}
		self.visited_states = []


	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		
		# initialize data structures
		if state not in self.visited_states:
			for opp_ac_ in self.possibleActions:
				self.Csa[(state,opp_ac_)] = 0.0
				for ac_ in self.possibleActions:
					self.Qvalues[(state,ac_,opp_ac_)] = self.initVals
			self.n[state] = 0.0
			self.visited_states += [state]

		if nextState not in self.visited_states:
			for opp_ac_ in self.possibleActions:
				self.Csa[(nextState,opp_ac_)] = 0.0
				for ac_ in self.possibleActions:
					self.Qvalues[(nextState,ac_,opp_ac_)] = self.initVals
			self.n[nextState] = 0.0
			self.visited_states += [nextState]
		
		self.cache = {
			'sCur' : state,
			'action' : action,
			'opp_action' : oppoActions[0],
			'reward'     : reward,
			'status'     : status,
			'sNex'  : nextState
		}

		

	
	def compute_V(self,state):
		s = [] # inner sums to take max over
		if self.n[state] == 0.0:
			return 0.0, np.random.choice(self.possibleActions)
		for action in self.possibleActions:
			Qs = [ float(self.Qvalues[(state,action,ac_)] * self.Csa[(state,ac_)]) for ac_ in self.possibleActions] # to be summed
			s += [1 / self.n[state] * sum(Qs)]
		v = max(s)
		act_max_i = np.random.choice([i for i,val in enumerate(s)  if val == v])
		action_max = self.possibleActions[act_max_i]
		return v, action_max

	def learn(self):
		sCur = self.cache['sCur']
		action = self.cache['action']
		sNex = self.cache['sNex']
		r    = self.cache['reward']
		a    = self.learningRate
		g    = self.discountFactor
		opp_action = self.cache['opp_action']
		trip = (sCur,action,opp_action)
		
		VsNex,_ = self.compute_V(sNex)

		tmp = self.Qvalues[trip]
		
		self.Qvalues[trip] = (1 - a) * self.Qvalues[trip] + a * (r + g * VsNex)
		
		self.Csa[(sCur,opp_action)] += 1
		self.n[sCur] += 1
		
		return self.Qvalues[trip] - tmp


	def act(self):
		if random.random() < self.epsilon or self.curState not in self.n.keys(): # e greedy
			action = np.random.choice(self.possibleActions)
		else:
			_, action = self.compute_V(self.curState)
			
		return action

	def setEpsilon(self, epsilon) :
		self.epsilon = epsilon 
		
	def setLearningRate(self, learningRate) :
		self.learningRate = learningRate


	def setState(self, state):
		self.curState = state

	def toStateRepresentation(self, rawState):
		if isinstance(rawState, str):
			return rawState
		else:
			b = list(rawState)
			c =[]
			for a in b: 
				c.extend(a)
			d = []
			for a in c:
				d.append(tuple(a))
			state = tuple(d)
			return state
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		lr = max(1 - episodeNumber/45000,1e-5)
		e  = 0.1
		if episodeNumber > 30000:
			e = 0.005
		if episodeNumber > 45000:
			e = 1e-6
		return lr,e

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0
	goals = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
			
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
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1
			
			if reward[0]>0:
				goals+=1

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx]
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation
		print('Episode', episode,'goals: ', goals)
