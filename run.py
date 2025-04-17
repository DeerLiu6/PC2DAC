import os
import sys
import numpy as np
import random
import tensorflow as tf
import datetime
import json
from matplotlib import pyplot as plt

from env import define
from env import environment
import mec_rl_with_uav

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger("logs/log.txt")
sys.stderr = Logger("logs/log.txt")

print("TensorFlow version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
plt.rcParams['figure.figsize'] = (9, 9)

map_size = 200
uav_num = 4
server_num = 4
sensor_num = 30

uav_obs_r = 60

uav_collect_r = 40
server_collect_r = 30

uav_move_r = 6
sensor_move_r = 3

rho = 1.225
C_D0 = 0.05
S = 0.1
g = 9.81
e0 = 0.8
A_R = 6
MAX_EPOCH = 10000
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.85
TAU = 0.8
BATCH_SIZE = 128
alpha = 0.9
beta = 0.1
Epsilon = 0.2
render_freq = 32

up_freq = 8
FL = True
FL_omega = 0.5

map_seed = 1
rand_seed = 17
np.random.seed(map_seed)
random.seed(map_seed)
tf.random.set_seed(rand_seed)

params = {
    'map_size': map_size,
    'uav_num': uav_num,
    'server_num': server_num,
    'sensor_num': sensor_num,
    'uav_obs_r': uav_obs_r,
    'uav_collect_r': uav_collect_r,
    'server_collect_r': server_collect_r,
    'uav_move_r': uav_move_r,
    'sensor_move_r': sensor_move_r,

    'MAX_EPOCH': MAX_EPOCH,
    'MAX_EP_STEPS': MAX_EP_STEPS,
    'LR_A': LR_A,
    'LR_C': LR_C,
    'GAMMA': GAMMA,
    'TAU': TAU,
    'BATCH_SIZE': BATCH_SIZE,
    'alpha': alpha,
    'beta': beta,
    'Epsilon': Epsilon,
    'learning_seed': rand_seed,
    'env_seed': map_seed,
    'up_freq': up_freq,
    'render_freq': render_freq,
    'FL': FL,
    'FL_omega': FL_omega
}

mec_world = define.MEC_world(map_size, uav_num, server_num, sensor_num, uav_obs_r, uav_collect_r, server_collect_r, uav_move_r, sensor_move_r)

env = environment.MEC_RL_ENV(mec_world)

MAAC = mec_rl_with_uav.MEC_RL_With_Uav(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE, Epsilon)

m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs('logs/hyperparam', exist_ok=True)
f = open('logs/hyperparam/%s.json' % m_time, 'w')

json.dump(params, f)
f.close()

MAAC.train(MAX_EPOCH, MAX_EP_STEPS, up_freq=up_freq, render=True, render_freq=render_freq, FL=FL, FL_omega=FL_omega)
sys.stdout.log.close()
sys.stderr.log.close()