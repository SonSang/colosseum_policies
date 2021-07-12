# TO DELETE : Simple code to add path...
import sys
import os
dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if dirname not in sys.path:
    sys.path.append(dirname)

import argparse

import tensorflow as tf

import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import set_global_seed, tournament, Logger
from texas_holdem_no_limit.rlcard_dqn_settings import *

SAVE_FILE_NAME = 'texas-holdem-no-limit-dqn'

def train(args):
    # Seed numpy, torch, random
    set_global_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    with tf.Session() as sess:
        # Initialize the DQN agent
        agent = DQNAgent(sess, 
                        'dqn',
                        action_num=DQN_ACTION_NUM,
                        state_shape=DQN_STATE_SHAPE,
                        mlp_layers=DQN_MLP_LAYERS)
        agents = [agent for i in range(env.player_num)]
        env.set_agents(agents)

        # Initialize variables in the graph
        global_step = tf.Variable(0, name="global_step", trainable=False)
        init = tf.global_variables_initializer()
        sess.run(init)

        for episode in range(args.num_episodes):
            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for pts in trajectories:
                for ts in pts:
                    agent.feed(ts)

        # Save model
        save_path = os.path.join(args.log_dir, SAVE_FILE_NAME)
        tf.compat.v1.train.Saver().save(sess, save_path)
        print('Model saved in', save_path)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN example in RLCard")
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='model/')

    args = parser.parse_args()
    args.env = 'no-limit-holdem'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)