# Local Distillation in Federated Learning for EEG-based Emotion Recognition

## How to use?

1. Prepare environment

a. Create new environment
```
conda create -n <ENV_NAME> python=3.9
conda activate <ENV_NAME>
```

b. Install pytorch >= 1.8.1, please refer to https://pytorch.org/get-started/locally/
```
# Conda
# please refer to https://pytorch.org/get-started/locally/
# e.g. CPU version
conda install pytorch==1.11.0 torchvision torchaudio cpuonly -c pytorch
# e.g. GPU version
conda install pytorch==1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

c. Install additional packages

```
pip install flwr==1.12.0 hydra-core==1.3.2 matplotlib==3.9.2 omegaconf==2.3.0 pandas==2.2.3 ray==2.38.0 SciencePlots==2.1.1 streamlit==1.39.0 torcheeg==1.1.2
pip install 'scipy<1.13'
```

2. Download dataset
* DEAP: Download the dataset from [here](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html) and put the `data_preprocessed_python` folder to `dataset/eeg_data/deap`
* SEED: Download the dataset from [here](https://bcmi.sjtu.edu.cn/home/seed/index.html) and put the `Preprocessed_EEG` folder to `dataset/eeg_data/seed`
* DREAMER: Download the dataset from [here](https://zenodo.org/records/546113#.Wi_vFjXSOGU) and put the `DREAMER.mat` file to `dataset/eeg_data/dreamer`

3. Run `python dataset.py` to prepare raw normalized and grid EEG data for each dataset from `dataset/eeg_data` to `dataset/processed_data`. You can download the processed data from [Google Drive](https://drive.google.com/file/d/1mcL0JF2bvbt_YthHOVs1Nh2u4rKezqph/view?usp=sharing).

4. Run `python pretrain.py` to pre-train teacher models for each dataset ([see below for more detail](#pretrain)). Make sure to put the models in `models/client`. If you don't want to train by yourself, you can download the pre-trained models from [Google Drive](https://drive.google.com/file/d/1Fe7MXpTxPCLPspKFSK5PN1xzvEPtACN4/view?usp=sharing).

5. Run `python main.py` to start federated training ([see below for more detail](#main)). You can set the configuration in `config/base.yaml`. Output will be saved in `output` folder.

6. Additionally, you can run `streamlit run predict.py` to run a Streamlit dashboard for inference using pre-trained models. You can also run the notebooks for visualization and analysis.

## Folder Structure

- **`config/base.yaml`**: Configuration file for `main.py`, used to specify experiment parameters and hyperparameters.
- **`dataset/eeg_data`**: Folder to store EEG datasets.
  - `deap`: Contains `data_preprocessed_python` folder.
  - `dreamer`: Contains `DREAMER.mat` file.
  - `seed`: Contains `SEED_EEG/Preprocessed_EEG` folder.
- **`dataset/processed_data`**: Stores processed EEG data by `dataset.py`.
- **`models/client`**: Contains pre-trained teacher models by `pretrain.py`.

## Main Modules

### 1. `main.py`: Federated Training <a id="main"></a>
This module performs federated training using the provided datasets and configuration.

#### Usage
Run with Hydra by specifying arguments in the format `[ARG1]=[VALUE] [ARG2]=[VALUE] ...`.

Examples:
1. **Assigning Client Resources**:
   ```
   python main.py num_cpus=8 num_gpus=1
   ```
   (See more about assigning resources at [Flower Documentation](https://flower.ai/docs/framework/how-to-run-simulations.html#assigning-client-resources))

2. **Running a Single Experiment** (overriding default config values):
   ```
   python main.py batch_size=32 num_rounds=10 config_fit.lr=0.001
   ```

3. **Running Multiple Experiments** (all combinations):
   ```
   python main.py --multirun seed=1,2,3,4,5 config_fit.mode=offline,online config_fit.temperature=1,5,10 config_fit.alpha=0,0.25,0.5,0.75,1
   ```

   *Notes: For complete hyperparameters, see `config/base.yaml`.*

### 2. `pretrain.py`: Pre-Training for Teacher Models <a id="pretrain"></a>
Separately pre-train teacher models for use in federated training.

#### Usage
```
python pretrain.py [-h] --seed_number SEED_NUMBER --dataset {deap,seed,dreamer} --overlap_percent {0,25,50,75} [--batch_size BATCH_SIZE] [--lr LR] [--test_ratio TEST_RATIO] [--epochs EPOCHS]
```

Example:
```
python pretrain.py --seed_number 1 --dataset deap --overlap_percent 0
```

To view all options:
```
python pretrain.py --help
```

### 3. `predict.py`: Inference Demo
A Streamlit dashboard for inference using pre-trained models.

#### Usage
```
streamlit run predict.py
```

## Supporting Modules

- **`client.py`**: Spawns virtual clients for federated training.
- **`dataset.py`**: Prepares local datasets for training and evaluation.
- **`distillation.py`**: Contains functions for local knowledge distillation.
- **`model.py`**: Manages the server-side model repository.
- **`server.py`**: Implements server aggregation strategy for federated learning.
- **`utils.py`**: Utility functions, including seed setting for reproducibility.

## Notebooks

- **`viz_representation.ipynb`**: Visualizes spatial and temporal representations of the pre-trained models.
- **`viz_with_experiment.ipynb`**: Displays tabular and graphical results post-experiment.
- **`viz_without_experiment.ipynb`**: Conducts exploratory data analysis (EDA) prior to training.
