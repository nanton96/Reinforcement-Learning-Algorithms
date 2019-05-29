#!/usr/bin/env python3
# encoding utf-8
import random
from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor = discountFactor
		self.epsilon = epsilon
		self.Qvalues = {}
		self.episode = []

		self.KperSA = {}

		self.visited_states = []
	def learn(self):
		G = 0
		Qestimates = []
		

		self.episode.reverse()
		for i,(state,action,reward) in enumerate(self.episode):
			
			if state not in self.visited_states:
				self.visited_states += [state]
				for ac_ in self.possibleActions:
					self.Qvalues[(state,ac_)] = 0
					self.KperSA[(state,ac_)] = 0
				vcur = 0
			else:
				vcur = self.Qvalues[(state,action)]
			
			previous_states = [(tup[:2]) for tup in self.episode[::-1]][:len(self.episode)-i-1]

			if (state,action) not in previous_states:
				
				self.KperSA[(state,action)] += 1
			
				
				G = self.discountFactor * G + reward
				# update mean with incremental approach
				vcur = vcur + 1/self.KperSA[(state,action)] * (G - vcur)
				self.Qvalues[(state,action)] = vcur
				Qestimates.append(vcur)
			
		return self.Qvalues, Qestimates[::-1]
				
	def toStateRepresentation(self, state):
		return tuple(state)

	def setExperience(self, state, action, reward, status, nextState):
		# storing the information for the episode which we will iterate over
		# in the learn() method
		self.episode.append((state,action,reward)) 

	def setState(self, state):
		self.curState = state

	def reset(self):
		self.observed = []
		self.episode = []

	def act(self):
		if random.random() < self.epsilon: # e greedy
			action = np.random.choice(self.possibleActions)
		else:
			action_vals = [self.Qvalues[(self.curState,ac_)] for ac_ in self.possibleActions]
			best_action_val = max(action_vals)
			actionIdx = np.random.choice([i for i,ac_val in enumerate(action_vals)  if ac_val == best_action_val])
			action = self.possibleActions[actionIdx]
		return action

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		e = max(1 - episodeNumber/4500,1e-6)
		return e

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=5000)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 0.2)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	goals = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):	
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		if status == 1:
				goals+=1
		
		agent.learn()
		print(episode ,':',goals)