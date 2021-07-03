Scripts to run trained [minigo](https://github.com/tensorflow/minigo) networks in PettingZoo environment.

1. Install requirements. Tensorflow has to be version 1.15.0 to use trained minigo weight files.
```
pip install -r requirements.txt
```
2. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
3. Run shell script to download trained minigo weights from [GCS](https://console.cloud.google.com/storage/browser/minigo-pub?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false). You can change version and name of the model in the script. Please refer to [minigo](https://github.com/tensorflow/minigo) to learn about the models.
```
./download_minigo_weights.sh
```
4. Run scripts and watch minigo play go in PettingZoo environment! There are two scripts to run minigo. Both files need network version and name of the network file to run.
* minigo_policies_mcts.py : Real minigo uses MCTS strategy to play go. This script lets minigo to use the strategy. Therefore, you can specify number of readouts to use in the strategy. However, it does not use observation of PettingZoo environment, as MCTS nodes store their own observation in themselves.
```
python minigo_policies_mcts.py --num_readouts=400 --network_version=17 --network_weights=./go/minigo-models/001003-leviathan
```
* minigo_policies_no_mcts.py : This script does not use MCTS strategy, but use observation of PettingZoo environment. It keeps track of the observations and feed them into the network to get policy, pi, and use it to sample next action.
```
python minigo_policies_no_mcts.py --network_version=17 --network_weights=./go/minigo-models/001003-leviathan
```