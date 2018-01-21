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


NOTES

MAX MATMUL AFTER 500 EPISODES 1857407.0
MIN MATMUL AFTER 500 EPISODES 749411.0

'''
import gym
import numpy as np
import collections
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
		
		self.bestMatmul = 1300000
		self.lowMatmul = 1100000
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
		matmul1=np.rint(np.matmul(observation,params))

		#Point = collections.namedtuple('Point', ['x', 'y'])
		#p = Point(1, y=2)
		#p.x, p.y
		#1 2
		#p[0], p[1]
		#1 2
		if matmul1>self.bestMatmul:
			self.bestMatmul=matmul1

		if matmul1<self.lowMatmul:
			self.lowMatmul=matmul1
		action=np.rint(np.matmul(observation,params))-749411.0
		
		max=(1857407.0-749411.0)/18 #max with everything is set to 255 is 8323200.0
		
		
		for i in range(18):
			if (action>(i*max)) and (action<((i+1)*max)):
				action = i
				break
		else: 
			action =0
			print("MATMUL OUT OF RANGE"+matmul1)
		return action
		
		#return 0 if np.matmul(observation, params) < (max)

	def episode(self, params):
		# Reset the environment
		#Point = collections.namedtuple('Point', ['x', 'y'])
		#p = Point(1, y=2)
		#p.x, p.y
		#1 2
		#p[0], p[1]
		#1 2
		observation = self.env.reset()
		
		score = 0

		# Repeatedly choose an action, apply the action, and update the score
		for i in range(5000): #how many actions you do in each trial, doesn't always run 5000 times if it tips over before it reaches then
			action = self.choose_action(observation, params) #inputting the rest function and the random number, getting back left or right
			observation, reward, done, info = self.env.step(action)  #step(self, action): Step the environment by one timestep. Returns observation, reward, done, info.
			#print(reward)
			if (reward>0):
				score += reward    #reward is difference between old score and new score

			if done: # The episode has ended
				print(score)
				break
				
				
		return score

	def run(self, linear_random=False):
		best_score = 0
		best_params = None

		
		
		# Start new episodes to find the best set of params
		for j in range(50):
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
		print(self.bestMatmul)
		print(self.lowMatmul)
if __name__ == '__main__':
    player = FishDerbySolver(monitor=True)
    player.run()