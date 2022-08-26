#!/usr/bin/env python
# coding: utf-8

# # conterfactual expalantion for DRL baced-on policy distillation

# ### import packages

# In[1]:


import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from policy_distillation import main as policy_distillation 
import random
from tkinter import Y
import numpy as np
import torch
import torch.nn as nn
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
# import matplotlib 
# matplotlib.use("Qt5Agg")
# from matplotlib import pyplot as plt
import gym
import highway_env
from matplotlib import pyplot as plt

from scipy.signal import convolve, gaussian
from copy import deepcopy
import os
import io
import base64
import time
import glob
from IPython.display import HTML
import local_lib as myLib

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from stable_baselines3 import SAC,A2C,DDPG, DQN

from itertools import count
from time import time, strftime, localtime
import gym
import scipy.optimize
from tensorboardX import SummaryWriter
from core.models import *
from core.agent_ray_pd import AgentCollection
from utils.utils import *
import numpy as np
import ray
import envs
from trpo import trpo
from student import Student
from teacher import Teacher
import os
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ",device)


# ### environment setup

# In[2]:
num_of_actions=5
num_Of_vehicles_Under_vision = 5
vehicle_attr = 4

low_dimention_env = gym.make("highway-fast-v0")
low_dimention_env.configure({
    "observation": {
        "type": "Kinematics",
        "vehicles_count": num_Of_vehicles_Under_vision,
        "features": ["x", "y", "vx", "vy"],
        "normalize": False,
        "absolute": True,
        # "order": "sorted"
    }
})
low_dimention_env.reset()

high_dimention_env = gym.make("highway-fast-v0")
high_dimention_env.configure({
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
})
high_dimention_env.reset()




# ## teacher training phase

# In[3]:


# model = myLib.DQN_creation(policy='MlpPolicy', env=env, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=5e-4,
#                 buffer_size=15000, learning_starts=200, batch_size=128, gamma=0.8, train_freq=1, gradient_steps=1,
#                 target_update_interval=50, exploration_fraction=0.7, verbose=1, tensorboard_log="highway_dqn/")
# model.learn(int(10e4))
fileName = "model"
# myLib.Save_DQN_model(model,fileName)
teacher_model = myLib.Load_DQN_model(fileName=fileName)


# ## student training phase

# In[5]:



import argparse

parser = argparse.ArgumentParser(description='Policy distillation')
# Network, env, MDP, seed
parser.add_argument('--hidden-size', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of hidden layers')
parser.add_argument('--env-name', default="highway-fast-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--load-models', default=True, action='store_true',
                    help='load_pretrained_models')


# Teacher policy training
parser.add_argument('--agent-count', type=int, default=10, metavar='N',
                    help='number of agents (default: 100)')
parser.add_argument('--num-teachers', type=int, default=1, metavar='N',
                    help='number of teacher policies (default: 1)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--cg-damping', type=float, default=1e-2, metavar='G',
                    help='damping for conjugate gradient (default: 1e-2)')
parser.add_argument('--cg-iter', type=int, default=10, metavar='G',
                    help='maximum iteration of conjugate gradient (default: 1e-1)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization parameter for critics (default: 1e-3)')
parser.add_argument('--teacher-batch-size', type=int, default=1000, metavar='N',
                    help='per-iteration batch size for each agent (default: 1000)')
parser.add_argument('--sample-batch-size', type=int, default=10000, metavar='N',
                    help='expert batch size for each teacher (default: 10000)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--device', type=str, default=device,
                    help='set the device (cpu or cuda)')
parser.add_argument('--num-workers', type=int, default=10,
                    help='number of workers for parallel computing')
parser.add_argument('--num-teacher-episodes', type=int, default=10, metavar='N',
                    help='num of teacher training episodes (default: 100)')

# Student policy training
parser.add_argument('--lr', type=float, default=1e-5, metavar='G',
                    help='adam learnig rate (default: 1e-3)')
parser.add_argument('--test-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--student-batch-size', type=int, default=1000, metavar='N',
                    help='per-iteration batch size for student (default: 1000)')
parser.add_argument('--sample-interval', type=int, default=10, metavar='N',
                    help='frequency to update expert data (default: 10)')
parser.add_argument('--testing-batch-size', type=int, default=10000, metavar='N',
                    help='batch size for testing student policy (default: 10000)')
parser.add_argument('--num-student-episodes', type=int, default=1000, metavar='N',
                    help='num of teacher training episodes (default: 1000)')
parser.add_argument('--loss-metric', type=str, default='kl',
                    help='metric to build student objective')
parser.add_argument('--algo', type=str, default='sgd',
                    help='update method')
parser.add_argument('--storm-interval', type=int, default=10, metavar='N',
                    help='frequency of storm (default: 10)')
parser.add_argument('--init-alpha', type=float, default=1.0, metavar='G',
                    help='storm init alpha (default: 1.0)')
args = parser.parse_args()



student = policy_distillation(args, low_dimention_env)


# ## potential Counterfactuals
    #################################for test only
env = gym.make("highway-fast-v0")
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": num_Of_vehicles_Under_vision,
        "features": ["x", "y", "vx", "vy"],
        "normalize": False,
        "absolute": True,
        # "order": "sorted"
    }
}
env.configure(config)

obs,done = env.reset(), False
for _ in range(20):
    while done == False:
        action = student.policy(torch.tensor(obs.reshape(obs.size), device=device)).loc.argmax().tolist()
        obs, reward, done, info = env.step(action)
        env.render()
    obs, done = env.reset(),False
env.close()
#################################end test
# ### second environment (lower dimensional input)

obs = env.reset()
env.render()

exp_is_valid = []
indx=np.array([1])
CFs, desireAction = local_lib.CF_find_2(env=env,student_model=student_model,obs=obs,device=device,indx=indx,
                      num_Of_vehicles_Under_vision=num_Of_vehicles_Under_vision,vehicle_attr=vehicle_attr)


# ## counterfactuals validation

# In[ ]:


image_env = DummyVecEnv([train_env])
video_length = 2 * image_env.envs[0].config["duration"]
image_env = VecVideoRecorder(image_env, "highway_cnn/videos/",
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="dqn-agent")

for i, cf in enumerate(CFs):
    obs_image = image_env.reset()
    obs_kinematics = myLib.CF2Env3(obs, cf, indx, env)
    for index, vehicles in enumerate(image_env.venv.envs[0].env.road.vehicles[0:num_Of_vehicles_Under_vision]):
        vehicles.position=cf[index,0:2]
        # vehicles.velocity=CF[indices,2:4]
#applying the CFs to environments
    image_env.venv.envs[0].env.render()
    image_env.venv.envs[0].env.render()
    image_env.venv.envs[0].env.render()
    obs_image=image_env.venv.envs[0].env.observation_type.observe() 
    ##prepare data
    plt.imshow(obs_image[3].T, cmap=plt.get_cmap('gray'))
    plt.show()
    action,_ = teacher_model.predict(obs_image)
    action_prim = student_model(torch.tensor(obs_kinematics.observation_type.observe()        .reshape(num_Of_vehicles_Under_vision*vehicle_attr), device=device)).argmax(dim=-1).item()
    # if action_prim==action_desire then the exp is valide
    if(action == action_prim == desireAction[i]):
        exp_is_valid.append(1)
    else:
        exp_is_valid.append(0)
    print(exp_is_valid)


# ## visulize the finded CFs

# In[ ]:




