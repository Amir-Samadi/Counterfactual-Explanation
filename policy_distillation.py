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
from stable_baselines3 import DQN
import highway_env
import local_lib as myLib
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
# torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_default_tensor_type('torch.FloatTensor')
dtype = torch.float
torch.set_default_dtype(dtype)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



'''
1. train single or multiple teacher policies
2. collect samples from teacher policy
3. use KL or W2 distance as metric to train student policy
4. test student policy
'''

def train_teachers():
    envs = []
    teacher_policies = []
    time_begin = time()
    print('Training {} teacher policies...'.format(args.num_teachers))
    for i in range(args.num_teachers):
        print('Training no.{} teacher policy...'.format(i + 1))
        env = gym.make(args.env_name)
        envs.append(env)
        teacher_policies.append(trpo(env, args))
    time_pretrain = time() - time_begin
    print('Training teacher is done, using time {}'.format(time_pretrain))
    return envs, teacher_policies

def main(args, env):
    ray.init(num_cpus=args.num_workers, num_gpus=1)
    # policy and envs for sampling
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    exp_date = strftime('%Y.%m.%d', localtime(time()))
    writer = SummaryWriter(log_dir='./exp_data/{}/{}_{}'.format(exp_date, args.env_name, time()))
    # load saved models if args.load_models
    if args.load_models:
        envs = []
        teacher_policies = []
        dummy_env = env
        dummy_env.env.reset()
        envs.append(dummy_env)
    else:
        envs, teacher_policies = train_teachers()

    dataset_df = pd.read_excel('DQN_CNN_regression.xlsx', index_col=0)
    numeric_feature_names = ['obs_0_x', 'obs_0_y', 'obs_0_vx',  'obs_0_vy',
                            'obs_1_x', 'obs_1_y', 'obs_1_vx',  'obs_1_vy',
                            'obs_2_x', 'obs_2_y', 'obs_2_vx',  'obs_2_vy',
                            'obs_3_x', 'obs_3_y', 'obs_3_vx',  'obs_3_vy',
                            'q_value_0','q_value_1','q_value_2','q_value_3','q_value_4']
    expert_data = []
    numeric_features = dataset_df[numeric_feature_names]
    tensor_numeric_features = torch.tensor(numeric_features.values, dtype=torch.float)
    for i,item in enumerate(tensor_numeric_features):
        expert_data.append([])
        expert_data[-1].append(item[0:16])
        expert_data[-1].append(item[16:21])
        expert_data[-1].append(torch.tensor([0.001,0.1,0.001,0.1,0.1], dtype=torch.float, device=args.device))


    student = Student(args)
    print('Training student policy...')
    time_beigin = time()
    # train student policy
    for iter in count(1):
        if args.algo == 'npg':
            loss = student.npg_train(expert_data)
        elif args.algo == 'storm':
            if iter == 1:
                loss, prev_params, prev_grad, direction = student.storm_train(None, None, None, expert_data, iter)
            else:
                loss, prev_params, prev_grad, direction = student.storm_train(prev_params, prev_grad, direction, expert_data, iter)
        else:
            loss = student.train(expert_data)
        writer.add_scalar('{} loss'.format(args.loss_metric), loss.data, iter)
        print('Itr {} {} loss: {:.2f}'.format(iter, args.loss_metric, loss.data))
        if iter > args.num_student_episodes:
            break
    time_train = time() - time_beigin
    print('Training student policy finished, using time {}'.format(time_train))
    return student



if __name__ == '__main__':
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
    parser.add_argument('--device', type=str, default='cpu',
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

    main(args)
