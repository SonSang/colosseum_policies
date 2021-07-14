Scripts to train [rlcard](https://github.com/datamllab/rlcard) networks and run them in PettingZoo environment.

1. Install requirements. Tensorflow has to be version 1.15.0 to use rlcard networks.
```
pip install -r requirements
```
2. Train rlcard network that implements DQN and save the trained network weights. You can give hyperparameters to use in the training. 
```
python rlcard_learn_dqn.py --num_episodes=5000 --log_dir=rlcard-models/
```
* rlcard_dqn_settings.py : You can change the layers used in the DQN in this file. Please modify DQN_MLP_LAYERS. It must stay invariant for the network used in the training, and the network we use in the PettingZoo environment for evaluation.

3. Run script and watch rlcard DQN play no-limit texas holdem in PettingZoo environment!
* rlcard_policies_dqn.py : This script requires name of the network file to run.
```
python rlcard_policies_dqn.py --network_weights=rlcard-models/texas-holdem-no-limit-dqn
```