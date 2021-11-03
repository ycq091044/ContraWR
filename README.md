# Code Scripts and Processing Files for EEG Sleep Staging Paper

### 1. Folder Tree
- ./src_preprocess (data preprocessing files for SHHS and Sleep EDF)
    - sleepEDF_cassette_process.py (script for processing Sleep EDF data)
    - shhs_processing.py (script for processing SHHS dataset)

- ./src
    - loss.py (the contrastive loss function of MoCo, SimCLR, BYOL, SimSiame and our ContraWR)
    - model.py (the encoder model for Sleep EDF and SHHS data)
    - self_supervised.py (the code for running self-supervised model)
    - supervised.py (the code for running supervised STFT CNN model)
    - utils.py (other functionalities, e.g., data loader)

### 2. Data Preparation
#### 2.1 Instructions for Sleep EDF
- Step1: download the Sleep EDF data from https://physionet.org/content/sleep-edfx/1.0.0/
    - we will use the Sleep EDF cassette portion
    ```python
    mkdir SLEEP_data; cd SLEEP_data
    wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
    ```
- Step2: running sleepEDF_cassette_process.py to process the data
    - running the following command line. The data will be stored in **./SLEEP_data/cassette_processed/pretext**, **./SLEEP_data/cassette_processed/train** and **./SLEEP_data/cassette_processed/test**
    ```python
    cd ../src_preprocess
    python sleepEDF_cassette_process.py
    ```

#### 2.2 Instructions for SHHS
- Step1: download the SHHS data from https://sleepdata.org/datasets/shhs
    ```python
    mkdir SHHS_data; cd SHHS_data
    [THEN DOWNLOAD YOUR DATASET HERE, NAME THE FOLDER "SHHS"]
    ```
- Step2: running shhs_preprocess.py to process the data
    - running the following command line. The data will be stored in **./SHHS_data/processed/pretext**, **./SHHS_data/processed/train** and **./SHHS_data/processed/test**
    ```python
    cd ../src_preprocess
    python shhs_process.py
    ```

### 3. Running the Experiments
First, go to the ./src directory, then run the **supervised model**
```python
cd ./src
# run on the SLEEP dataset
python -W ignore supervised.py --dataset SLEEP --n_dim 128
# run on the SHHS dataset
python -W ignore supervised.py --dataset SHHS --n_dim 256
```

Second, run the self-supervised models
```python
# run on the SLEEP dataset
python -W ignore self_supervised.py --dataset SLEEP --model ContraWR --n_dim 128
# run on the SHHS dataset
python -W ignore self_supervised.py --dataset SHHS --model ContraWR --n_dim 256
# try other self-supervised models
# change "ContraWR" to "MoCo", "SimCLR", "BYOL", "SimSiam"
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
If you find this repo is useful, please cite our paper. Feel free to contact me <chaoqiy2@illinois.edu> for any problem.
