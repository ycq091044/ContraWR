# Open EEG Data Preprocessing and SSL Baselines
This repo provides 
- Open data source **SHHS** Processing
- Open data source **Sleep-EDF** Processing
- **self-supervised EEG learning baselines**: MoCo, SimCLR, SimSiame, BYOL, inclduing our new ContraWR, check out in ArXiv https://arxiv.org/abs/2110.15278

### 1. Folder Tree
- ```./preprocess``` (data preprocessing files for SHHS and Sleep EDF)
    - **sleepEDF_cassette_process.py** (script for processing Sleep EDF data)
    - **shhs_processing.py** (script for processing SHHS dataset)

- ```./src```
    - **loss.py** (the contrastive loss function of MoCo, SimCLR, BYOL, SimSiame and our ContraWR)
    - **model.py** (the encoder model for Sleep EDF and SHHS data)
    - **self_supervised.py** (the code for running self-supervised model)
    - **supervised.py** (the code for running supervised STFT CNN model)
    - **utils.py** (other functionalities, e.g., data loader)

### 2. Data Preparation
#### 2.1 Instructions for Sleep EDF
- Step1: download the Sleep EDF data from https://physionet.org/content/sleep-edfx/1.0.0/
    - we will use the Sleep EDF cassette portion
    ```python
    # create the data folder and enter
    mkdir SLEEP_data; cd SLEEP_data
    wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
    ```
- Step2: running **sleepEDF_cassette_process.py** to process the data
    - running the following command line. The data will be stored in **./pretext**, **./train** and **./test**
    ```python
    # enter this folder and run preprocessing
    cd ../preprocess
    python sleepEDF_cassette_process.py --windowsize 30 --multiprocess 8
    ```
- Here, ```windowsize``` means how long is each "signal epoch", usually it is 30 seconds, ```multiprocess``` means how many process will be used. The same below.

#### 2.2 Instructions for SHHS
- Step1: download the SHHS data from https://sleepdata.org/datasets/shhs (you probability need certificates first)
    ```python
    # create the data folder and enter
    mkdir SHHS_data; cd SHHS_data
    [THEN DOWNLOAD YOUR DATASET HERE, NAME THE DATA FOLDER "SHHS"]
    ```
- Step2: running shhs_preprocess.py to process the data
    - running the following command line. The data will be stored in **./pretext**, **./train** and **./test**
    ```python
    # enter this folder and run preprocessing
    cd ../src_preprocess
    python shhs_process.py --windowsize 30 --multiprocess 8
    ```
- Here, ```windowsize``` means how long is each "signal epoch", usually it is 30 seconds, ```multiprocess``` means how many process will be used. The same below.

### 3. Running the Experiments
#### 3.1 supervised model
```python
cd ./src
# run on the SLEEP-EDF dataset
python -W ignore supervised.py --dataset SLEEP --n_dim 128
# run on the SHHS dataset
python -W ignore supervised.py --dataset SHHS --n_dim 256
```

#### 3.2 run the self-supervised learning model
```python
# run on the SLEEP-EDF dataset
python -W ignore self_supervised.py --dataset SLEEP --model ContraWR --n_dim 128
# run on the SHHS dataset
python -W ignore self_supervised.py --dataset SHHS --model ContraWR --n_dim 256
# try other self-supervised models: "MoCo", "SimCLR", "BYOL", "SimSiam"
```

### Citation
```bibtex
@article{yang2021self,
  title={Self-supervised EEG Representation Learning for Automatic Sleep Staging},
  author={Yang, Chaoqi and Xiao, Danica and Westover, M Brandon and Sun, Jimeng},
  journal={arXiv preprint arXiv:2110.15278},
  year={2021}
}
```
If you find this repo is useful, please cite our paper. Feel free to contact me <chaoqiy2@illinois.edu> or send an issue for any problem.

### Clarification on Bandpass Filtering
The intuition is that the low-pass signals and high-pass signals might be both useful. So a broader idea is to maintain either the low-frequency or high-frequency or both low-and-high frequency information for data augmentation. My primary thinking is to design a low-pass filter (a, b) and a high-pass filter (c, d) for each dataset, where a < b < c < d.
 
Theoretically, these four values are hyperparameters and need to be set based on the validation set. Here, in our paper, the values are set more in an ad-hoc way since the datasets are fairly large and it is impossible to run a grid search for a perfect (a, b, c, d) combination. So what I did is first choose a combination and get the validation results. Based on the val results and some intuitions, we refine the combination and get the new validation results again and finally converge to the current values.

