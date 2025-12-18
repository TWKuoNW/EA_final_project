import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # dino_wm/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math
import time
import gymnasium as gym
import gym_pusht
import torch
from DinoV2Encoder import DinoV2Encoder

class Evaluator:
	enc = DinoV2Encoder(device="cuda:0")
	selfCost = None #cls.selfCost[state[i]]
	selfDamage = None #cls.selfDamage[state[i]]
	EnhanceDamage = None #cls.EnhanceDamage[state[i-1]][state[i]]
	cfg = None # Add cfg as a class variable

	@classmethod
	def ObjFunc(cls, state, render= False):
		# Evaluate the sequence of actions in the PushT environment
		env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
		env_goal = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")

		fixed_state = [20.0, 250, 100.0, 200.0, 0.0]  # agent_x, agent_y, block_x, block_y, angle
		goal_state = [320.30368666, 217.29127243, 253.99707306, 253.41529713, 0.75337255]
		
		env.reset(options={"reset_to_state": fixed_state})
		env_goal.reset(options={"reset_to_state": goal_state})

		rewardEnd = 0.0
		# Each action consists of two consecutive values in the individual
		for i in range(len(state) // 2):
			action = state[2 * i : 2 * i + 2]
			observation, reward, terminated, truncated, info = env.step(action)
			rewardEnd = reward  # keep last reward
			if(render == True):
				env.render()
			if terminated or truncated:
				observation, info = env.reset(options={"reset_to_state": fixed_state})
		
		emb = cls.enc.encode(env.render())
		emb_goal = cls.enc.encode(env_goal.render())
		emb_distance = torch.norm(emb - emb_goal, p=2).item()
		# print("Embedding distance to goal:", emb_distance)
		# print("embedding shape:", emb.shape)
		env.close()

		# Normalized number of steps
		nSeqSteps = float(len(state)) / float(cls.cfg.nGenesCfg)
		#nSeqSteps = len(state) // 2

		objectives = [nSeqSteps, -emb_distance]

		return objectives

