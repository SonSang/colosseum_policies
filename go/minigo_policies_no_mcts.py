# TO DELETE : Simple code to add path...
import sys
import os
dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if dirname not in sys.path:
    sys.path.append(dirname)

# Script
from pettingzoo.classic import go_v3
from pettingzoo.classic.go import go
from absl import app, flags
from go.minigo import dual_net
from go.minigo import utils
from go.minigo import coords
from go.minigo import symmetries
import numpy as np
import random

# Define command line arguments
flags.DEFINE_integer(
    "network_version",
    None,
    "Minigo network version.",
    9,
    17
)
flags.DEFINE_string(
    "network_weights",
    None,
    "Minigo network weights file."
)
FLAGS = flags.FLAGS

# Get Go board by integrating black and white plane
def go_board(black_plane, white_plane):
    return black_plane * go.BLACK + white_plane * go.WHITE

# Extract stone features from past 8 boards : [16, N, N]
def stone_features(last_boards, to_play: int):
    last_eight = np.zeros([8, go.N, go.N], dtype=np.int8)
    for i in range(8):
        last_eight[i] = last_boards[-1 - i] if i < len(last_boards) else last_boards[0]

    features = np.zeros([16, go.N, go.N], dtype=np.uint8)
    features[::2] = last_eight == to_play
    features[1::2] = last_eight == -to_play
    return np.rollaxis(features, 0, 3)

# Extract color feature from current player
def color_to_play_feature(to_play: int):
    if to_play == go.BLACK:
        return np.ones([go.N, go.N, 1], dtype=np.uint8)
    else:
        return np.zeros([go.N, go.N, 1], dtype=np.uint8)

# Extract input features from past boards and current player info
def input_features(last_boards, to_play: int):
    return np.concatenate([stone_features(last_boards, to_play), color_to_play_feature(to_play)], axis=2)

# Set network flags according to the specified version
def set_network_version(version: int) -> None:
    if version is 17:
        # V17 uses SeNet instead of ResNet
        dual_net.FLAGS.use_SE_bias = True
        dual_net.FLAGS.use_SE = True
    elif version is 16:
        # V16 uses 40 ResNet blocks, intead of 20
        dual_net.FLAGS.trunk_layers = 39

def main(argv) -> None:
    del argv # Unused

    # Initialize network and player
    set_network_version(FLAGS.network_version)
    try:
        with utils.logged_timer("Loading weights from %s ..." % FLAGS.network_weights):
            network = dual_net.DualNetwork(FLAGS.network_weights)
    except Exception as e:
        print("=============== Error while loading network weights.")
        print("=============== Error :", e)

    # Play the game
    env = go_v3.env()
    env.reset()

    last_boards = []
    for agent in env.agent_iter():
        _, reward, done, info = env.last()
        if done:
            print("Result :", env.agents[0], "=", env.rewards[env.agents[0]])
            print("Result :", env.agents[1], "=", env.rewards[env.agents[1]])
            break
        observation = _['observation']
        action_mask = _['action_mask']

        # Get planes
        to_play = go.BLACK if agent is env.agents[0] else go.WHITE
        black_plane = observation[:,:,0] if to_play is go.BLACK else observation[:,:,1]
        white_plane = observation[:,:,1] if to_play is go.BLACK else observation[:,:,0]
        
        # Extract features to feed into the network
        board = go_board(black_plane, white_plane)
        last_boards.append(board)
        processed = [input_features(last_boards, to_play)]

        # Preprocess input features
        if FLAGS.use_random_symmetry:
            syms_used, processed = symmetries.randomize_symmetries_feat(
                processed)
        
        # Run the network
        outputs = network.sess.run(network.inference_output,
                                feed_dict={network.inference_input: processed})
        probabilities, value = outputs['policy_output'], outputs['value_output']

        # Postprocess results
        if FLAGS.use_random_symmetry:
            probabilities = symmetries.invert_symmetries_pi(
                syms_used, probabilities)

        # Select action from CDF
        probabilities = probabilities[0]
        probabilities[np.where(action_mask == 0)] = 0.0
        cdf = np.cumsum(probabilities)
        cdf /= cdf[-1]
        selection = random.random()
        fmove = cdf.searchsorted(selection)
        move = coords.from_flat(fmove)

        # Make a move
        env.step(fmove)

        # Show current plane
        print(agent, "Played >>", coords.to_gtp(move))
        env.render()

    env.close()

if __name__=="__main__":
    flags.mark_flag_as_required("network_version")
    flags.mark_flag_as_required("network_weights")
    app.run(main)