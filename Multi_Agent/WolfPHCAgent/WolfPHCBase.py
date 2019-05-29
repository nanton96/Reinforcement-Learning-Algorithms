#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
from copy import deepcopy
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta, loseDelta):
		super(WolfPHCAgent, self).__init__()
		
		self.winDelta = winDelta
		self.loseDelta = loseDelta
		self.discountFactor = discountFactor
		self.initVals = 0.0
		self.Qvalues = {}
		self.learningRate = learningRate
		self.main_policy = {}
		self.average_policy = {}
		self.C = {}
		self.cache = {}
		self.visited_states = []

	def setExperience(self, state, action, reward, status, nextState):
		if nextState not in self.visited_states:
			self.visited_states.append(nextState)
			self.C[nextState] = 0
			for ac_ in self.possibleActions:
				self.Qvalues[(nextState,ac_)] = self.initVals
				self.main_policy[(nextState,ac_)] = 1 / len(self.possibleActions)
				self.average_policy[(nextState,ac_)] = 0

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
		# if random.random() < self.epsilon: # e greedy
		# 	action = np.random.choice(self.possibleActions)
		# else:
		# 	action_vals = [self.Qvalues[(self.curState,ac_)] for ac_ in self.possibleActions]
		# 	best_action_val = max(action_vals)
		# 	actionIdx = np.random.choice([i for i,ac_ in enumerate(action_vals)  if ac_ == best_action_val])
		# 	action = self.possibleActions[actionIdx]

		action = np.random.choice(self.possibleActions,p = [self.main_policy[(self.curState,ac_)] for ac_ in self.possibleActions])


		return action

	def calculateAveragePolicyUpdate(self):
		out = []
		self.C[self.curState] +=1
		for ac_ in self.possibleActions:
			pair = (self.curState,ac_)
			self.average_policy[pair] += 1 / self.C[self.curState] * (self.main_policy[pair] - self.average_policy[pair])
			out.append(self.average_policy[pair])
		return out

	def calculatePolicyUpdate(self):
		
		optimal_actions, subobptimal_actions = self.find_suboptimal()
		p_moved = 0
		delta = self.which_delta()
		for ac_ in subobptimal_actions:
			p_moved += min(delta/len(subobptimal_actions),self.main_policy[(self.curState,ac_)])
			self.main_policy[(self.curState,ac_)] -= min(delta/len(subobptimal_actions),self.main_policy[(self.curState,ac_)])
		out = []
		for ac_ in optimal_actions:
			self.main_policy[(self.curState,ac_)] += p_moved / (len(self.possibleActions) - len(subobptimal_actions))
		for ac_ in self.possibleActions:
			out.append(self.main_policy[(self.curState,ac_)])
		return out

	def which_delta(self):

		sum_p = 0.0
		sum_av_p = 0.0
		for ac_ in self.possibleActions:
			pair = (self.curState,ac_)
			sum_p += self.main_policy[pair] * self.Qvalues[pair]
			sum_av_p += self.average_policy[pair] * self.Qvalues[pair]

		if sum_p >= sum_av_p:
			return self.winDelta
		else:
			return self.loseDelta

		
	def find_suboptimal(self):

		Qmax = max([self.Qvalues[(self.curState,ac_)] for ac_ in self.possibleActions])
		if self.curState not in self.visited_states:
			return self.possibleActions, []
		else:	
			a_opt = [ac_ for ac_ in self.possibleActions if self.Qvalues[(self.curState,ac_)] == Qmax]
			a_subopt = [ac_ for ac_ in self.possibleActions if self.Qvalues[(self.curState,ac_)] != Qmax]

			return a_opt,a_subopt

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


	def setState(self, state):
		if state not in self.visited_states:
			self.visited_states.append(state)
			self.C[state] = 0
			for ac_ in self.possibleActions:
				self.Qvalues[(state,ac_)] = self.initVals
				self.main_policy[(state,ac_)] = 1 / len(self.possibleActions)
				self.average_policy[(state,ac_)] = 0
		self.curState = state

	def setWinDelta(self, winDelta):
		self.winDelta = winDelta
	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta
	def setLearningRate(self,learningRate):
		self.learningRate = learningRate

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		
		self.winDelta=0.0001
		self.loseDelta=0.001
		if episodeNumber>10000:
			self.winDelta=0.001
			self.loseDelta=0.01
		if episodeNumber>20000:
			self.winDelta=0.01
			self.loseDelta=.1
		if episodeNumber>30000:
			self.winDelta=0.05
			self.loseDelta=.5
		if episodeNumber>40000:
			self.winDelta=0.1
			self.loseDelta=1
		self.learningRate = max((1-episodeNumber/50000),1e-5)*0.5
		return self.loseDelta, self.winDelta, self.learningRate


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=100000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	goals = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		
		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1
			
			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1
			if reward[0]==1:
				goals+=1
			observation = nextObservation
		print('Episode', episode,'goals: ', goals)