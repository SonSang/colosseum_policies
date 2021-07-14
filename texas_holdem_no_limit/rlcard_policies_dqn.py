# TO DELETE : Simple code to add path...
import sys
import os
dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if dirname not in sys.path:
    sys.path.append(dirname)

# Script
from pettingzoo.classic import texas_holdem_no_limit_v3
from absl import app, flags

import tensorflow as tf

from texas_holdem_no_limit.rlcard_dqn_settings import *

# Define command line arguments
flags.DEFINE_string(
    "network_weights",
    None,
    "rlcard texas-holdem-no-limits DQN weights file."
)
FLAGS = flags.FLAGS

def main(argv) -> None:
    del argv # Unused

    env = texas_holdem_no_limit_v3.env()
    
    with tf.Session() as sess:
        # Initialize agent
        from rlcard.agents import DQNAgent
        network = DQNAgent(sess, 
                        'dqn',
                        action_num=DQN_ACTION_NUM,
                        state_shape=DQN_STATE_SHAPE,
                        mlp_layers=DQN_MLP_LAYERS)
                            
        try:
            tf.compat.v1.train.Saver().restore(sess, FLAGS.network_weights)
        except Exception as e:
            print("=============== Error while loading network weights.")
            print("=============== Error :", e)
            raise

        # Play the game
        env.reset()
        for agent in env.agent_iter():
            _, reward, done, info = env.last()
            if done:
                for agent in env.agents:
                    print("Result: ", agent, "=", env.rewards[agent])
                break
            observation = _['observation']
            action_mask = _['action_mask']

            state = {}
            state['obs'] = observation
            state['legal_actions'] = [i for i in range(len(action_mask)) if action_mask[i]]
            
            move, _ = network.eval_step(state)
            
            # Make a move
            env.step(move)

            # Show
            print(agent, "Played >>", Action(move))
            env.render()

        env.close()

if __name__=="__main__":
    flags.mark_flag_as_required("network_weights")
    app.run(main)