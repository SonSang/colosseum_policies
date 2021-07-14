from enum import Enum

# Do not change these
DQN_ACTION_NUM = 6
DQN_STATE_SHAPE = [54]

# Change this to alter DQN layers
DQN_MLP_LAYERS = [64, 64]

# Action Names
class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_HALF_POT = 3
    RAISE_POT = 4
    ALL_IN = 5