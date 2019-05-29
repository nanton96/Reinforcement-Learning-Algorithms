from MDP import MDP

class BellmanDPSolver(object):
	def __init__(self,discountRate):
		self.MDP = MDP()
		self.gamma = discountRate
		self.initVs()
	def initVs(self):
		self.values = {s: 0 for s in self.MDP.S}
		self.policy = {s: self.MDP.A for s in self.MDP.S}
	
	def BellmanUpdate(self):
		
		for s in self.MDP.S:
			best_v = -10**20
			best_a = []
			n_value = {a:0 for a in self.MDP.A}
			for a in self.MDP.A:
				
				for s_ in self.MDP.probNextStates(s,a).keys():
					n_value[a] += self.MDP.probNextStates(s,a)[s_] * (self.MDP.getRewards(s,a,s_) + self.gamma * self.values[s_])
				if n_value[a] > best_v:
					best_v = n_value[a]
			
			self.values[s] = best_v
			for a in self.MDP.A:
				if n_value[a] == best_v:
					best_a += [a]
			self.policy[s] = best_a
		return self.values, self.policy
		
if __name__ == '__main__':
	solution = BellmanDPSolver(0.9)
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)
