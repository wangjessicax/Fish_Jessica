'''
A model-free solver for the CartPole environment in the OpenAI Gym.

Every episode, the player chooses chooses four random weights and chooses
to move left or right based on the product of multiplying those weights
with the observation. The player receives a score for the episode and keeps
track of the weights from the episode which received the highest score.

This example serves as a scaffolding for approaching a gym environment.

Written by Emily Pries based on Kevin Frans' post at http://kvfrans.com/simple-algoritms-for-solving-cartpole/


reset(self): Reset the environment's state. Returns observation.
step(self, action): Step the environment by one timestep. Returns observation, reward, done, info.
render(self, mode='human', close=False): Render one frame of the environment. The default mode will do something human friendly, such as pop up a window. Passing the close flag signals the renderer to close any such windows.

Couldn't figure out step number 3 or 4
'''
import gym
import numpy as np
#self is the object that is calling it; in java, it is this


#GRADIENT DESCENT NUMPY


class FishDerbySolver():
	# Solver initialization
	def __init__(self, monitor=False):
		# Make the gym environment
		self.env = gym.make('FishingDerby-ram-v0')
		self.env.reset()
		'''
		for _ in range(1000):
			self.env.render()
			self.env.step(self.env.action_space.sample()) # take a random action
		

		for i_episode in range(20):
			observation = self.env.reset()
			for t in range(100):
				self.env.render()
				print(observation)
				action = self.env.action_space.sample()
				observation, reward, done, info = self.env.step(action)
				if done:
					print("Episode finished after {} timesteps".format(t+1))
					break

		'''
		print(self.env.action_space)
		print(self.env.observation_space)
		# Set environment variables
		self.MAX_SCORE = 500 # The best score possible for this environment is 500

		# Store videos of the episodes in the "cartpole-random" directory - default is no video
		if monitor:
			self.env = gym.wrappers.Monitor(self.env, './fishing-derby', force=True)

	def choose_action(self, observation, params):
		# Cart pole lets us move left (action = 0) or right (action = 1)
		# This moves left if observation x params < 0, otherwise move right
		#return 0 if np.matmul(observation, params) < 0 else 1
		#print("this is the matmul") #max matmul is 8323200.0 min matmul is 0
		#print(np.matmul(observation,params))

		#return np.rint(np.matmul(observation,params))
		action=np.rint(np.matmul(observation,params))-1000000
		max=(1500000-1000000)/18 #max with everything is set to 255 is 8323200.0
		if action < (max):
			action = 0
			#print("action was made 0")
		elif (action>(max)) and (action<(2*max)):
			action=1
			#print("action was made 1")
		elif (action>(2*max)) and (action<(3*max)):
			action=2
			#print("action was made 2")
		elif (action>(3*max)) and (action<(4*max)):
			action=3
			#print("action was made 3")
		elif (action>(4*max)) and (action<(5*max)):
			action=4
			#print("action was made 4")
		elif (action>(5*max)) and (action<(6*max)):
			action=5
			#print("action was made 5")
		elif (action>(6*max)) and (action<(7*max)):
			action=6
			#print("action was made 6")
		elif (action>(7*max)) and (action<(8*max)):
			action=7
			#print("action was made 7")
		elif (action>(8*max)) and (action<(9*max)):
			action=8
			#print("action was made 8")
		elif (action>(9*max)) and (action<(10*max)):
			action=9
			#print("action was made 9")
		elif (action>(10*max)) and (action<(11*max)):
			action=10
			#print("action was made 10")
		elif (action>(11*max)) and (action<(12*max)):
			action=11
			#print("action was made 11")
		elif (action>(12*max)) and (action<(13*max)):
			action=12
			#print("action was made 12")
		elif (action>(13*max)) and (action<(14*max)):
			action=13
			#print("action was made 13")
		elif (action>(14*max)) and (action<(15*max)):
			action=14
			#print("action was made 14")
		elif (action>(15*max)) and (action<(16*max)):
			action=15
			#print("action was made 15")
		elif (action>(16*max)) and (action<(17*max)):
			action=16
			#print("action was made 16")
		elif (action>(17*max)) and (action<(8323200.0)):
			action=17
			#print("action was made 17")
		return action
		#return 0 if np.matmul(observation, params) < (max)

	def episode(self, params):
		# Reset the environment
		observation = self.env.reset()
		
		score = 0

		# Repeatedly choose an action, apply the action, and update the score
		for i in range(5000): #how many actions you do in each trial, doesn't always run 5000 times if it tips over before it reaches then
			action = self.choose_action(observation, params) #inputting the rest function and the random number, getting back left or right
			observation, reward, done, info = self.env.step(action)  #step(self, action): Step the environment by one timestep. Returns observation, reward, done, info.
			print(reward)
			score += reward    #reward is difference between old score and new score
			
			if done: # The episode has ended
				print(score)
				print("THIS EPISODE HAS ENDED")
				break
				
				
		return score

	def run(self, linear_random=False):
		best_score = 0
		best_params = None
		

		# Start new episodes to find the best set of params
		for j in range(100000):
			# Randomly assign four weights and run an episode
			params = np.random.rand(128)*255
			print("Episode {}: {}".format(j, best_score))
			score = self.episode(params) #calling episode, returns score

			# Keep track of the best scoring params
			if score > best_score:
				best_score = score
				best_params = params
				print("Episode {}: {}".format(j, best_score))

			if score == self.MAX_SCORE: # We can't find a better score, so stop
				break
		print("Best score was from Episode {}: {}".format(j, best_score))

if __name__ == '__main__':
    player = FishDerbySolver(monitor=True)
    player.run()