# TO DELETE : Simple code to add path...
import sys
import os
dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if dirname not in sys.path:
    sys.path.append(dirname)

# Script
from pettingzoo.classic import go_v3
from absl import app, flags
from go.minigo import dual_net
from go.minigo import utils
from go.minigo import coords
from go.minigo.strategies import MCTSPlayer

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
# From strategies.py
flags.declare_key_flag('num_readouts')

FLAGS = flags.FLAGS

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
        sys.exit()
    player = MCTSPlayer(network, resign_threshold=-1.0)     # Disable resign
    player.initialize_game()

    # Must run this once at the start to expand the root node
    first_node = player.root.select_leaf()
    prob, val = network.run(first_node.position)
    first_node.incorporate_results(prob, val, first_node)

    # Play the game
    readouts = FLAGS.num_readouts   # Defined in strategies.py
    env = go_v3.env()
    env.reset()
    for agent in env.agent_iter():
        # Observe is not needed when we use "player"
        _, reward, done, info = env.last()
        if done:
            print("Result :", env.agents[0], "=", env.rewards[env.agents[0]])
            print("Result :", env.agents[1], "=", env.rewards[env.agents[1]])
            break

        # Choose next action based on MCTS technique
        player.root.inject_noise()
        current_readouts = player.root.N
        # We want to do "X additional readouts", rather than "up to X readouts"
        while player.root.N < current_readouts + readouts:
            player.tree_search()
        move = player.pick_move()
        fmove = coords.to_flat(move)

        # Make a move
        player.play_move(move)
        env.step(fmove)

        # Show current plane
        print(agent, "Played >>", coords.to_gtp(move))
        env.render()

    env.close()

if __name__=="__main__":
    flags.mark_flag_as_required("network_version")
    flags.mark_flag_as_required("network_weights")
    app.run(main)